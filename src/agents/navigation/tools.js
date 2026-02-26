import { tool } from 'ai';
import { z } from 'zod';

export const navigateTool = tool({
    description: 'Navigate to a page in the application (under construction)',
    parameters: z.object({
        destination: z.string().describe('The page or section to navigate to'),
    }),
    execute: async ({ destination }) => {
        return {
            success: false,
            message: `Navigation to "${destination}" is under construction. Lazy devs are working on it!`
        };
    },
});
