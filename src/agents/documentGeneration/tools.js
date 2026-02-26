import { tool } from 'ai';
import { z } from 'zod';

export const generateDocumentTool = tool({
    description: 'Generate project documents (under construction)',
    parameters: z.object({
        documentType: z.enum(['serial_letter', 'rea', 'monthly_narrative']).describe('Type of document to generate'),
        sourceId: z.string().optional().describe('Source RFI, story, or project ID'),
    }),
    execute: async ({ documentType, sourceId }) => {
        return {
            success: false,
            message: `Generating ${documentType} is under construction. Lazy devs are on it!`
        };
    },
});
