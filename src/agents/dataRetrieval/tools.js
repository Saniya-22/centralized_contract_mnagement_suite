import { tool } from 'ai';
import { z } from 'zod';

export const queryDataTool = tool({
    description: 'Query platform data (companies, opportunities, projects) - under construction',
    parameters: z.object({
        dataType: z.enum(['companies', 'opportunities', 'projects', 'solicitations', 'jobs', 'account']).describe('Type of data to query'),
        query: z.string().describe('The search query'),
    }),
    execute: async ({ dataType, query }) => {
        return {
            success: false,
            message: `Querying ${dataType} for "${query}" is under construction. Lazy devs are on it!`
        };
    },
});
