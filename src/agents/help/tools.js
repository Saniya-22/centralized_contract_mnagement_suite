import { tool } from 'ai';
import { z } from 'zod';
import { ragService } from '../../services/rag.service.js';

export const searchRegulationsTool = tool({
    description: `Search government contracting regulations (FAR, DFARS, EM 385-1-1) for relevant clauses and requirements.

Use this tool when users ask about:
- Federal Acquisition Regulation (FAR) clauses
- Defense Federal Acquisition Regulation Supplement (DFARS)
- Safety and Health Requirements (EM 385-1-1)
- Contract clauses, requirements, or compliance questions
- Specific regulation topics like small business, construction, cybersecurity, etc.

Examples:
- "differing site conditions clause" -> searches for FAR 52.236-2
- "cybersecurity requirements for defense contracts" -> searches DFARS
- "fall protection requirements" -> searches EM 385-1-1`,
    parameters: z.object({
        query: z.string().describe('Natural language search query about regulations'),
        regulationType: z.enum(['FAR', 'DFARS', 'EM385', 'ALL']).optional()
            .describe('Filter by regulation type: FAR, DFARS, EM385, or ALL (default)'),
        partNumber: z.string().optional()
            .describe('Filter by part number (e.g., "19", "36", "52" for FAR)')
    }),
    execute: async ({ query, regulationType, partNumber }) => {
        try {
            const results = await ragService.retrieveRegulationChunks({
                query,
                regulationType: regulationType === 'ALL' ? undefined : regulationType,
                partNumber,
                k: 8,
                tokenLimit: 16000
            });

            if (results.chunks.length === 0) {
                return {
                    success: false,
                    message: `No regulation content found for: "${query}"`,
                    suggestions: [
                        'Try broader search terms',
                        'Specify FAR, DFARS, or EM385 if you know the source',
                        'Search for specific clause numbers if known'
                    ]
                };
            }

            return {
                success: true,
                query,
                totalFound: results.totalFound,
                returned: results.returned,
                regulationType: regulationType || 'ALL',
                context: results.formattedContext,
                chunks: results.chunks.map(c => ({
                    source: c.metadata?.source,
                    part: c.metadata?.part,
                    preview: (c.text || '').substring(0, 300) + '...',
                    clauseReferences: c.metadata?.clause_references || [],
                    retrievalMethod: c.retrieval_methods?.join('+') || c.chunk_type || 'unknown',
                    score: c.rrf_score ?? c.similarity ?? null
                }))
            };
        } catch (error) {
            console.error('Error in searchRegulationsTool:', error);
            return {
                success: false,
                message: `Error searching regulations: ${error.message}`,
                error: true
            };
        }
    },
});

export const getClauseByReferenceTool = tool({
    description: `Retrieve a specific regulation clause by its reference number.

Use this tool when users ask for a specific clause like:
- "What is FAR 52.236-2?"
- "Show me DFARS 252.204-7012"
- "Get the text of FAR 52.212-4"

The tool will return the clause text and surrounding context.`,
    parameters: z.object({
        clauseReference: z.string()
            .describe('The clause reference (e.g., "FAR 52.236-2", "DFARS 252.204-7012", "EM 385 Section 05.A")')
    }),
    execute: async ({ clauseReference }) => {
        try {
            const result = await ragService.getClauseByReference(clauseReference);

            if (!result.found) {
                return {
                    success: false,
                    clauseReference,
                    message: result.context,
                    suggestions: [
                        'Verify the clause number is correct',
                        'Try searching with searchRegulations for related content',
                        'The clause may be in a different part of the regulation'
                    ]
                };
            }

            return {
                success: true,
                clauseReference,
                clause: result.clause,
                context: result.context
            };
        } catch (error) {
            console.error('Error in getClauseByReferenceTool:', error);
            return {
                success: false,
                clauseReference,
                message: `Error retrieving clause: ${error.message}`,
                error: true
            };
        }
    },
});

export const searchHelpTool = tool({
    description: 'Search platform help documentation and tutorials (coming soon)',
    parameters: z.object({
        query: z.string().describe('Help search query'),
        context: z.string().optional().describe('Current page or feature context'),
    }),
    execute: async ({ query, context }) => {
        return {
            success: false,
            message: `Platform help for "${query}" is coming soon. For now, use searchRegulations for FAR/DFARS/EM 385 questions.`,
            availableFeatures: [
                'searchRegulations - Search FAR, DFARS, and EM 385-1-1',
                'getClauseByReference - Get specific clause text'
            ]
        };
    },
});

export const helpAgentTools = {
    searchRegulations: searchRegulationsTool,
    getClauseByReference: getClauseByReferenceTool,
    searchHelp: searchHelpTool
};
