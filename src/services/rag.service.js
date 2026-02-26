import pg from 'pg';
import axios from 'axios';
import { OpenAI } from 'openai';

const {
    OPENAI_API_KEY,
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS,
    PG_DENSE_TABLE,
    PG_SSLMODE,
    EMBEDDING_MODEL_URL, EMBEDDING_MODEL_API_KEY, EMBEDDING_MODEL_NAME_OS,
    BGE_RERANKER_ENDPOINT
} = process.env;

const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

const ssl = PG_SSLMODE === 'require' ? { rejectUnauthorized: false } : false;

/** RRF constant (k=60 is standard) */
const RRF_K = 60;

class RagService {
    constructor() {
        this.pool = new pg.Pool({
            host: PG_HOST,
            port: parseInt(PG_PORT || '5432'),
            database: PG_DB,
            user: PG_USER,
            password: PG_PASS,
            ssl,
        });
    }

    // ─── Embeddings ───────────────────────────────────────────────

    async getDenseEmbedding(text) {
        try {
            if (!EMBEDDING_MODEL_URL && openai) {
                const response = await openai.embeddings.create({
                    model: 'text-embedding-ada-002',
                    input: text
                });
                return response.data[0].embedding;
            }

            const response = await axios.post(
                `${EMBEDDING_MODEL_URL}/embeddings`,
                {
                    model: EMBEDDING_MODEL_NAME_OS,
                    input: text
                },
                {
                    headers: {
                        'Authorization': `Bearer ${EMBEDDING_MODEL_API_KEY}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
            return response.data.data[0].embedding;
        } catch (error) {
            console.error('[RAG] Error getting dense embedding:', error.message);
            throw error;
        }
    }

    // ─── Reranker ─────────────────────────────────────────────────
    // TODO: Replace GPT-4o-mini reranker with Cohere Rerank API for better speed and cost
    // https://docs.cohere.com/reference/rerank

    async rerank(query, chunks) {
        if (!chunks.length) return [];

        if (BGE_RERANKER_ENDPOINT) {
            return this._rerankBGE(query, chunks);
        }

        if (openai) {
            return this._rerankOpenAI(query, chunks);
        }

        return chunks.sort((a, b) => (b.rrf_score || b.similarity || 0) - (a.rrf_score || a.similarity || 0));
    }

    async _rerankOpenAI(query, chunks) {
        const numbered = chunks.map((c, i) => {
            const preview = (c.text || '').substring(0, 300).replace(/\n/g, ' ');
            return `[${i}] ${preview}`;
        }).join('\n\n');

        try {
            const response = await openai.chat.completions.create({
                model: 'gpt-4o-mini',
                temperature: 0,
                messages: [
                    {
                        role: 'system',
                        content: 'You are a relevance scoring system. Given a query and numbered document excerpts, return ONLY a JSON array of objects with "index" (number) and "score" (0-10 relevance). Score 10 = highly relevant, 0 = irrelevant. No explanation.'
                    },
                    {
                        role: 'user',
                        content: `Query: "${query}"\n\nDocuments:\n${numbered}`
                    }
                ],
                response_format: { type: 'json_object' }
            });

            const content = response.choices[0]?.message?.content || '';
            const parsed = JSON.parse(content);
            const scores = parsed.scores || parsed.results || parsed;

            const scoreArr = Array.isArray(scores) ? scores : [];
            const scoreMap = new Map();
            for (const item of scoreArr) {
                if (typeof item.index === 'number' && typeof item.score === 'number') {
                    scoreMap.set(item.index, item.score);
                }
            }

            const ranked = chunks.map((chunk, i) => ({
                ...chunk,
                rerank_score: scoreMap.get(i) ?? 0
            }));
            ranked.sort((a, b) => b.rerank_score - a.rerank_score);

            console.log(`[RAG] Reranked ${chunks.length} chunks via GPT-4o-mini`);
            ranked.slice(0, 5).forEach((c, i) => {
                const preview = (c.text || '').substring(0, 80).replace(/\n/g, ' ');
                console.log(`[RAG]   ${i + 1}. rerank_score=${c.rerank_score} "${preview}..."`);
            });

            return ranked;
        } catch (error) {
            console.error('[RAG] Error in OpenAI reranking:', error.message);
            return chunks.sort((a, b) => (b.rrf_score || b.similarity || 0) - (a.rrf_score || a.similarity || 0));
        }
    }

    async _rerankBGE(query, chunks) {
        const candidateTexts = chunks.map(c => `Content from regulation document: ${c.text}`);

        try {
            const response = await axios.post(
                BGE_RERANKER_ENDPOINT,
                {
                    model: "BAAI/bge-reranker-base",
                    text_1: query,
                    text_2: candidateTexts
                }
            );

            const scores = response.data.data.map(d => d.score);
            const indices = scores.map((score, index) => ({ score, index }))
                .sort((a, b) => b.score - a.score)
                .map(item => item.index);

            return indices.map(i => chunks[i]);
        } catch (error) {
            console.error('[RAG] Error in BGE reranking:', error.message);
            return chunks;
        }
    }

    // ─── Dense (Vector) Search ────────────────────────────────────

    async denseSearch(embedding, k, namespace, metadataContains) {
        const vectorLiteral = JSON.stringify(embedding);
        let query = `
      SELECT id, namespace, text, metadata, (1 - (embedding <=> $1::vector)) AS similarity
      FROM ${PG_DENSE_TABLE}
      WHERE 1=1
    `;
        const params = [vectorLiteral];
        let paramIdx = 2;

        if (namespace) {
            query += ` AND namespace = $${paramIdx}`;
            params.push(namespace);
            paramIdx++;
        }

        if (metadataContains) {
            query += ` AND metadata @> $${paramIdx}::jsonb`;
            params.push(JSON.stringify(metadataContains));
            paramIdx++;
        }

        query += ` ORDER BY embedding <=> $1::vector LIMIT $${paramIdx}`;
        params.push(k);

        const result = await this.pool.query(query, params);
        return result.rows.map(row => ({ ...row, similarity: parseFloat(row.similarity), chunk_type: 'dense' }));
    }

    // ─── Full-Text Search (replaces BM25) ─────────────────────────

    _buildOrTsquery(text) {
        // Split into words, remove short/empty tokens, join with OR operator
        const terms = text
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(t => t.length > 1);
        if (!terms.length) return null;
        return terms.join(' | ');
    }

    async ftsSearch(queryText, k, namespace, metadataContains) {
        const orExpr = this._buildOrTsquery(queryText);
        if (!orExpr) return [];

        let sql = `
      SELECT id, namespace, text, metadata,
             ts_rank_cd(search_vector, query, 32) AS rank
      FROM ${PG_DENSE_TABLE},
           to_tsquery('english', $1) AS query
      WHERE search_vector @@ query
    `;
        const params = [orExpr];
        let paramIdx = 2;

        if (namespace) {
            sql += ` AND namespace = $${paramIdx}`;
            params.push(namespace);
            paramIdx++;
        }

        if (metadataContains) {
            sql += ` AND metadata @> $${paramIdx}::jsonb`;
            params.push(JSON.stringify(metadataContains));
            paramIdx++;
        }

        sql += ` ORDER BY rank DESC LIMIT $${paramIdx}`;
        params.push(k);

        const result = await this.pool.query(sql, params);
        return result.rows.map(row => ({ ...row, similarity: parseFloat(row.rank), chunk_type: 'fts' }));
    }

    // ─── Reciprocal Rank Fusion ───────────────────────────────────

    _reciprocalRankFusion(denseChunks, ftsChunks) {
        const scoreMap = new Map();
        const chunkMap = new Map();

        // Score from dense results (ranked by vector similarity)
        denseChunks.forEach((chunk, rank) => {
            const current = scoreMap.get(chunk.id) || 0;
            scoreMap.set(chunk.id, current + 1 / (RRF_K + rank + 1));
            if (!chunkMap.has(chunk.id)) {
                chunkMap.set(chunk.id, { ...chunk, retrieval_methods: ['dense'] });
            }
        });

        // Score from FTS results (ranked by ts_rank_cd)
        ftsChunks.forEach((chunk, rank) => {
            const current = scoreMap.get(chunk.id) || 0;
            scoreMap.set(chunk.id, current + 1 / (RRF_K + rank + 1));
            if (!chunkMap.has(chunk.id)) {
                chunkMap.set(chunk.id, { ...chunk, retrieval_methods: ['fts'] });
            } else {
                chunkMap.get(chunk.id).retrieval_methods.push('fts');
            }
        });

        // Build fused list sorted by RRF score
        const fused = [];
        for (const [id, rrfScore] of scoreMap) {
            const chunk = chunkMap.get(id);
            fused.push({ ...chunk, rrf_score: rrfScore });
        }
        fused.sort((a, b) => b.rrf_score - a.rrf_score);

        return fused;
    }

    // ─── Token estimation ─────────────────────────────────────────

    _estimateTokens(text) {
        const wordCount = (text || '').split(/\s+/).filter(Boolean).length;
        return Math.ceil(wordCount * 1.3);
    }

    // ─── Clause reference filtering ───────────────────────────────

    _filterByClauseReference(chunks, clauseReference) {
        if (!clauseReference) return chunks;

        const normalizedRef = clauseReference.toLowerCase().replace(/\s+/g, '');
        // Also try with dots and dashes for partial matches (e.g., "52.236" matches "52.236-2")
        const refParts = clauseReference.replace(/\s+/g, '').split(/[\.\-]/);

        const filtered = chunks.filter(chunk => {
            const text = (chunk.text || '').toLowerCase();
            const refs = chunk.metadata?.clause_references || [];

            // Direct text match (normalized)
            if (text.includes(normalizedRef)) return true;

            // Match against clause_references metadata
            if (refs.some(ref => {
                if (!ref.clause) return false;
                const refClause = ref.clause.toLowerCase().replace(/\s+/g, '');
                return refClause.includes(normalizedRef) || normalizedRef.includes(refClause);
            })) return true;

            // Partial match: if all parts of the reference appear in the text
            if (refParts.length > 1 && refParts.every(part => text.includes(part.toLowerCase()))) return true;

            return false;
        });

        return filtered.length > 0 ? filtered : chunks;
    }

    // ─── Logging helpers ──────────────────────────────────────────

    _logSearchResults(label, chunks, limit = 5) {
        console.log(`[RAG] ${label}: ${chunks.length} results`);
        chunks.slice(0, limit).forEach((chunk, i) => {
            const source = chunk.metadata?.source || '?';
            const part = chunk.metadata?.part || '?';
            const score = chunk.similarity?.toFixed(4) ?? chunk.rrf_score?.toFixed(4) ?? 'N/A';
            const preview = (chunk.text || '').substring(0, 120).replace(/\n/g, ' ');
            console.log(`[RAG]   ${i + 1}. [${source} Part ${part}] score=${score} type=${chunk.chunk_type} "${preview}..."`);
        });
    }

    // ─── Hybrid retrieval (client documents) ──────────────────────

    async retrieveHybridChunks({
        query,
        k = 5,
        clientId,
        documentType,
        policyNumber,
        tokenLimit = 12000
    }) {
        const denseEmb = await this.getDenseEmbedding(query);

        const metadataContains = {};
        if (documentType) metadataContains.classification = documentType;
        if (policyNumber) metadataContains.policy_number = policyNumber;

        const metaFilter = Object.keys(metadataContains).length ? metadataContains : null;

        const [denseChunks, ftsChunks] = await Promise.all([
            this.denseSearch(denseEmb, k, clientId, metaFilter),
            this.ftsSearch(query, k, clientId, metaFilter)
        ]);

        this._logSearchResults('Dense (hybrid)', denseChunks);
        this._logSearchResults('FTS (hybrid)', ftsChunks);

        const fusedChunks = this._reciprocalRankFusion(denseChunks, ftsChunks);

        this._logSearchResults('RRF fused (hybrid)', fusedChunks);

        const rerankedChunks = await this.rerank(query, fusedChunks);

        let currentTokens = 0;
        const finalContextChunks = [];

        for (const chunk of rerankedChunks) {
            const tokens = this._estimateTokens(chunk.text);
            if (currentTokens + tokens > tokenLimit) break;
            finalContextChunks.push(chunk);
            currentTokens += tokens;
        }

        return finalContextChunks.map(chunk => {
            const title = chunk.metadata?.document_path?.split('/').pop()?.split('.')[0] || "Document";
            return `Draft: ${title}\nContent: ${chunk.metadata?.relevant_text || chunk.text}`;
        }).join("\n----------\n");
    }

    // ─── Regulation retrieval ─────────────────────────────────────

    async retrieveRegulationChunks({
        query,
        regulationType,
        partNumber,
        clauseReference,
        k = 10,
        tokenLimit = 16000
    }) {
        const namespace = 'public-regulations';

        const metadataContains = {};
        if (regulationType) metadataContains.source = regulationType;
        if (partNumber) metadataContains.part = partNumber;

        const metaFilter = Object.keys(metadataContains).length ? metadataContains : null;

        console.log(`[RAG] retrieveRegulationChunks: query="${query}" type=${regulationType || 'ALL'} part=${partNumber || 'ALL'} clause=${clauseReference || 'none'}`);

        const denseEmb = await this.getDenseEmbedding(query);

        const [denseChunks, ftsChunks] = await Promise.all([
            this.denseSearch(denseEmb, k, namespace, metaFilter),
            this.ftsSearch(query, k, namespace, metaFilter)
        ]);

        this._logSearchResults('Dense (regulations)', denseChunks);
        this._logSearchResults('FTS (regulations)', ftsChunks);

        const fusedChunks = this._reciprocalRankFusion(denseChunks, ftsChunks);

        this._logSearchResults('RRF fused (regulations)', fusedChunks);

        const rerankedChunks = await this.rerank(query, fusedChunks);

        const filteredChunks = this._filterByClauseReference(rerankedChunks, clauseReference);

        let currentTokens = 0;
        const finalChunks = [];

        for (const chunk of filteredChunks) {
            const tokens = this._estimateTokens(chunk.text);
            if (currentTokens + tokens > tokenLimit) continue; // skip, try smaller chunks
            finalChunks.push(chunk);
            currentTokens += tokens;
        }

        console.log(`[RAG] Final: ${finalChunks.length} chunks, ~${currentTokens} tokens`);

        const formattedContext = finalChunks.map(chunk => {
            const source = chunk.metadata?.source || 'Unknown';
            const part = chunk.metadata?.part ? ` Part ${chunk.metadata.part}` : '';
            const refs = chunk.metadata?.clause_references || [];
            const clauseInfo = refs.length > 0
                ? `\nClauses Referenced: ${refs.map(r => `${r.type} ${r.clause}`).join(', ')}`
                : '';

            return `**Source: ${source}${part}**${clauseInfo}\n\n${chunk.text}`;
        }).join("\n\n---\n\n");

        return {
            chunks: finalChunks,
            formattedContext,
            totalFound: fusedChunks.length,
            returned: finalChunks.length
        };
    }

    // ─── Direct text search (exact clause number lookup) ─────────

    async directClauseSearch(clauseNum, k, namespace, source) {
        let sql = `
      SELECT id, namespace, text, metadata,
             position(lower($1) in lower(text)) AS match_pos
      FROM ${PG_DENSE_TABLE}
      WHERE text ILIKE $2
    `;
        const pattern = `%${clauseNum}%`;
        const params = [clauseNum, pattern];
        let paramIdx = 3;

        if (namespace) {
            sql += ` AND namespace = $${paramIdx}`;
            params.push(namespace);
            paramIdx++;
        }

        if (source) {
            sql += ` AND metadata->>'source' = $${paramIdx}`;
            params.push(source);
            paramIdx++;
        }

        // Order by match position — chunks starting with the clause number come first
        sql += ` ORDER BY match_pos ASC LIMIT $${paramIdx}`;
        params.push(k);

        const result = await this.pool.query(sql, params);
        return result.rows.map(row => ({ ...row, similarity: 1.0, chunk_type: 'direct' }));
    }

    // ─── Direct clause lookup ─────────────────────────────────────

    async getClauseByReference(clauseReference) {
        const match = clauseReference.match(/(FAR|DFARS|EM\s*385)\s*(\d+[\.\-][\d\-]+)/i);

        if (!match) {
            console.log(`[RAG] getClauseByReference: regex failed for "${clauseReference}", trying FTS fallback`);
            const ftsResults = await this.ftsSearch(clauseReference, 5, 'public-regulations', null);
            if (ftsResults.length > 0) {
                this._logSearchResults('FTS fallback (clause lookup)', ftsResults);
                const primaryChunk = ftsResults[0];
                return {
                    found: true,
                    clause: {
                        reference: clauseReference,
                        source: primaryChunk.metadata?.source,
                        part: primaryChunk.metadata?.part,
                        text: primaryChunk.text,
                        relatedClauses: primaryChunk.metadata?.clause_references || []
                    },
                    context: `**Source: ${primaryChunk.metadata?.source || 'Unknown'}${primaryChunk.metadata?.part ? ` Part ${primaryChunk.metadata.part}` : ''}**\n\n${primaryChunk.text}`
                };
            }

            return {
                found: false,
                clause: null,
                context: `Could not parse clause reference: ${clauseReference}. Expected format: "FAR 52.236-2" or "DFARS 252.204-7012"`
            };
        }

        const source = match[1].toUpperCase().replace(/\s+/g, '');
        const clauseNum = match[2];

        // First: try direct text match (most reliable for exact clause numbers)
        const directResults = await this.directClauseSearch(clauseNum, 10, 'public-regulations', source === 'EM385' ? 'EM385' : source);
        this._logSearchResults('Direct text search (clause lookup)', directResults);

        if (directResults.length > 0) {
            // Already sorted by match position (SQL ORDER BY match_pos ASC)
            const formattedContext = directResults.slice(0, 5).map(chunk => {
                const src = chunk.metadata?.source || 'Unknown';
                const part = chunk.metadata?.part ? ` Part ${chunk.metadata.part}` : '';
                return `**Source: ${src}${part}**\n\n${chunk.text}`;
            }).join("\n\n---\n\n");

            const primaryChunk = directResults[0];
            return {
                found: true,
                clause: {
                    reference: clauseReference,
                    source: primaryChunk.metadata?.source,
                    part: primaryChunk.metadata?.part,
                    text: primaryChunk.text,
                    relatedClauses: primaryChunk.metadata?.clause_references || []
                },
                context: formattedContext
            };
        }

        // Fallback: full retrieval pipeline
        console.log(`[RAG] getClauseByReference: direct search found nothing, falling back to full pipeline`);
        const results = await this.retrieveRegulationChunks({
            query: `${source} ${clauseNum}`,
            regulationType: source === 'EM385' ? 'EM385' : source,
            clauseReference: clauseNum,
            k: 5,
            tokenLimit: 16000
        });

        if (results.chunks.length === 0) {
            return {
                found: false,
                clause: null,
                context: `No results found for clause ${clauseReference}. This clause may not be in the indexed documents.`
            };
        }

        const primaryChunk = results.chunks[0];

        return {
            found: true,
            clause: {
                reference: clauseReference,
                source: primaryChunk.metadata?.source,
                part: primaryChunk.metadata?.part,
                text: primaryChunk.text,
                relatedClauses: primaryChunk.metadata?.clause_references || []
            },
            context: results.formattedContext
        };
    }
}

export const ragService = new RagService();
