import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as mupdf from 'mupdf';
import pg from 'pg';
import { OpenAI } from 'openai';
import natural from 'natural';
import murmurhash from 'murmurhash';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const NAMESPACE = 'public-regulations';
const CHUNK_SIZE = 2048;
const CHUNK_OVERLAP = 200;
const EMBEDDING_MODEL = 'text-embedding-ada-002';
const BATCH_SIZE = 20;

const {
    OPENAI_API_KEY,
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS,
    PG_DENSE_TABLE, PG_SPARSE_TABLE, PG_SSLMODE
} = process.env;

if (!OPENAI_API_KEY) {
    console.error('Error: OPENAI_API_KEY is required');
    process.exit(1);
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const ssl = PG_SSLMODE === 'require' ? { rejectUnauthorized: false } : false;
const pool = new pg.Pool({
    host: PG_HOST,
    port: parseInt(PG_PORT || '5432'),
    database: PG_DB,
    user: PG_USER,
    password: PG_PASS,
    ssl,
});

const denseTable = PG_DENSE_TABLE || 'embeddings_dense';
const sparseTable = PG_SPARSE_TABLE || 'embeddings_sparse';

const STOP_WORDS = new Set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself",
    "me", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours", "yourself", "yourselves"
]);

const stemmer = natural.PorterStemmer;
const tokenizer = new natural.WordTokenizer();

function encodeBM25(text) {
    if (!text) return { indices: [], values: [] };
    const tokens = tokenizer.tokenize(text.toLowerCase());
    const filtered = tokens.filter(t => !STOP_WORDS.has(t) && t.length > 1);
    const stemmed = filtered.map(t => stemmer.stem(t));
    const counts = {};
    for (const t of stemmed) counts[t] = (counts[t] || 0) + 1;
    const indices = [], values = [];
    for (const [term, count] of Object.entries(counts)) {
        indices.push(murmurhash.v3(term) | 0);
        values.push(count);
    }
    return { indices, values };
}

function extractTextFromBlock(block) {
    let text = '';
    for (const line of block.lines || []) {
        if (line.text) {
            text += line.text + '\n';
        } else if (line.spans) {
            for (const span of line.spans) {
                text += span.text || '';
            }
            text += '\n';
        }
    }
    return text;
}

function extractPageContent(page, pageNum) {
    const content = { pageNum, text: '', tables: [], hasImages: false };

    try {
        const stext = page.toStructuredText('preserve-whitespace,preserve-images');
        const json = stext.asJSON();
        const parsed = typeof json === 'string' ? JSON.parse(json) : json;

        for (const block of parsed.blocks || []) {
            if (block.type === 'text') {
                const blockText = extractTextFromBlock(block);
                if (blockText.trim()) {
                    content.text += blockText;
                }
            } else if (block.type === 'image') {
                content.hasImages = true;
            }
        }

        const lines = content.text.split('\n');
        let tableLines = [];
        for (const line of lines) {
            if ((line.match(/\t/g) || []).length >= 2 || (line.match(/\s{3,}/g) || []).length >= 2) {
                tableLines.push(line);
            } else if (tableLines.length >= 3) {
                content.tables.push(tableLines.join('\n'));
                tableLines = [];
            } else {
                tableLines = [];
            }
        }
        if (tableLines.length >= 3) {
            content.tables.push(tableLines.join('\n'));
        }

    } catch (error) {
        try {
            content.text = page.toStructuredText().asText();
        } catch (e) {
            console.warn(`  Page ${pageNum}: extraction failed`);
        }
    }

    return content;
}

function extractMetadata(filename, filePath) {
    const metadata = {
        filename,
        document_path: filePath,
        classification: 'regulation',
        indexed_at: new Date().toISOString()
    };

    if (filename.startsWith('FAR_')) {
        metadata.source = 'FAR';
        const match = filename.match(/FAR_(\d+)/);
        if (match) metadata.part = match[1];
    } else if (filename.startsWith('DFARS_')) {
        metadata.source = 'DFARS';
        if (filename.includes('Appendix')) {
            const match = filename.match(/Appendix_([A-Z])/);
            if (match) metadata.part = `Appendix_${match[1]}`;
        } else if (filename.includes('270')) {
            metadata.part = '270';
        } else {
            metadata.part = '201-253';
        }
    } else if (filename.includes('EM 385') || filename.includes('EM_385')) {
        metadata.source = 'EM385';
        metadata.part = '1-1';
    }

    return metadata;
}

function extractClauseReferences(text) {
    const refs = [];
    const farMatch = text.matchAll(/FAR\s+(\d+\.\d+(?:-\d+)?)/gi);
    for (const m of farMatch) refs.push({ type: 'FAR', clause: m[1] });
    const dfarsMatch = text.matchAll(/DFARS\s+(\d+\.\d+(?:-\d+)?)/gi);
    for (const m of dfarsMatch) refs.push({ type: 'DFARS', clause: m[1] });
    return refs;
}

function createChunks(fullText, chunkSize = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
    const chunks = [];
    if (!fullText || fullText.length === 0) return chunks;

    const paragraphs = fullText.split(/\n\n+/).filter(p => p.trim());

    let currentChunk = '';

    for (const para of paragraphs) {
        const trimmed = para.trim();
        if (!trimmed) continue;

        if (currentChunk.length + trimmed.length + 2 > chunkSize && currentChunk) {
            chunks.push(currentChunk.trim());
            const overlapStart = Math.max(0, currentChunk.length - overlap);
            currentChunk = currentChunk.slice(overlapStart) + '\n\n' + trimmed;
        } else {
            currentChunk += (currentChunk ? '\n\n' : '') + trimmed;
        }
    }

    if (currentChunk.trim()) {
        chunks.push(currentChunk.trim());
    }

    return chunks;
}

async function generateEmbeddings(texts) {
    const embeddings = [];
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
        const batch = texts.slice(i, i + BATCH_SIZE);
        process.stdout.write(`\r  Embeddings: ${i + batch.length}/${texts.length}`);

        try {
            const response = await openai.embeddings.create({
                model: EMBEDDING_MODEL,
                input: batch
            });
            for (const item of response.data) {
                embeddings.push(item.embedding);
            }
        } catch (error) {
            console.error(`\n  Embedding error: ${error.message}`);
            for (let j = 0; j < batch.length; j++) embeddings.push(null);
        }

        if (i + BATCH_SIZE < texts.length) {
            await new Promise(r => setTimeout(r, 50));
        }
    }
    console.log('');
    return embeddings;
}

async function storeChunks(chunks, embeddings, baseMetadata) {
    const client = await pool.connect();
    try {
        await client.query('BEGIN');

        for (let i = 0; i < chunks.length; i++) {
            const text = chunks[i];
            const embedding = embeddings[i];
            if (!embedding) continue;

            const refs = extractClauseReferences(text);
            const metadata = {
                ...baseMetadata,
                chunk_index: i,
                clause_references: refs,
                relevant_text: text.substring(0, 500)
            };

            const id = `${NAMESPACE}-${baseMetadata.filename}-${i}`;

            await client.query(
                `INSERT INTO ${denseTable} (id, namespace, text, metadata, embedding)
                 VALUES ($1, $2, $3, $4, $5::vector)
                 ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding`,
                [id, NAMESPACE, text, JSON.stringify(metadata), JSON.stringify(embedding)]
            );

            const sparseEmb = encodeBM25(text);
            await client.query(
                `INSERT INTO ${sparseTable} (id, namespace, text, metadata, embedding)
                 VALUES ($1, $2, $3, $4, $5::jsonb)
                 ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding`,
                [id, NAMESPACE, text, JSON.stringify(metadata), JSON.stringify(sparseEmb)]
            );
        }

        await client.query('COMMIT');
    } catch (error) {
        await client.query('ROLLBACK');
        throw error;
    } finally {
        client.release();
    }
}

async function clearNamespace() {
    console.log(`Clearing '${NAMESPACE}' namespace...`);
    try {
        await pool.query(`DELETE FROM ${denseTable} WHERE namespace = $1`, [NAMESPACE]);
        await pool.query(`DELETE FROM ${sparseTable} WHERE namespace = $1`, [NAMESPACE]);
    } catch (error) {
        console.warn('Warning:', error.message);
    }
}

async function processPDF(filePath) {
    const filename = path.basename(filePath, '.pdf');
    console.log(`\n>> ${filename}`);

    try {
        const buffer = fs.readFileSync(filePath);
        const doc = mupdf.Document.openDocument(buffer, 'application/pdf');
        const pageCount = doc.countPages();
        console.log(`  Pages: ${pageCount}`);

        const metadata = extractMetadata(filename, filePath);

        let fullText = '';
        let tableCount = 0;
        let imageCount = 0;

        for (let i = 0; i < pageCount; i++) {
            const page = doc.loadPage(i);
            const content = extractPageContent(page, i + 1);
            fullText += content.text + '\n\n';
            tableCount += content.tables.length;
            if (content.hasImages) imageCount++;

            if ((i + 1) % 100 === 0) {
                process.stdout.write(`\r  Extracted: ${i + 1}/${pageCount} pages`);
            }
        }
        if (pageCount >= 100) console.log('');

        console.log(`  Text: ${fullText.length} chars, Tables: ${tableCount}, Pages with images: ${imageCount}`);

        const chunks = createChunks(fullText);
        console.log(`  Chunks: ${chunks.length}`);

        if (chunks.length === 0) {
            console.log('  SKIPPED: No chunks generated');
            return { filename, chunks: 0, success: false };
        }

        const embeddings = await generateEmbeddings(chunks);

        await storeChunks(chunks, embeddings, metadata);
        console.log(`  Stored: ${chunks.length} chunks`);

        return { filename, chunks: chunks.length, success: true };

    } catch (error) {
        console.error(`  ERROR: ${error.message}`);
        return { filename, chunks: 0, success: false, error: error.message };
    }
}

async function main() {
    console.log('='.repeat(60));
    console.log('Government Regulations Ingestion');
    console.log('='.repeat(60));

    const specsDir = path.join(__dirname, '..', '..', 'specifications');
    if (!fs.existsSync(specsDir)) {
        console.error(`Specs directory not found: ${specsDir}`);
        process.exit(1);
    }

    const pdfFiles = [];
    for (const file of fs.readdirSync(specsDir)) {
        if (file.endsWith('.pdf')) pdfFiles.push(path.join(specsDir, file));
    }
    const dfarsDir = path.join(specsDir, 'DFARS');
    if (fs.existsSync(dfarsDir)) {
        for (const file of fs.readdirSync(dfarsDir)) {
            if (file.endsWith('.pdf')) pdfFiles.push(path.join(dfarsDir, file));
        }
    }

    console.log(`Found ${pdfFiles.length} PDFs`);
    await clearNamespace();

    const results = [];
    for (const pdf of pdfFiles) {
        results.push(await processPDF(pdf));
    }

    console.log('\n' + '='.repeat(60));
    console.log('SUMMARY');
    console.log('='.repeat(60));
    let total = 0, success = 0;
    for (const r of results) {
        const status = r.success ? 'OK' : 'FAIL';
        console.log(`${status.padEnd(6)} ${r.filename.padEnd(40)} ${r.chunks} chunks`);
        if (r.success) { total += r.chunks; success++; }
    }
    console.log('='.repeat(60));
    console.log(`Total: ${success}/${results.length} files, ${total} chunks`);

    await pool.end();
}

main().catch(e => { console.error(e); process.exit(1); });
