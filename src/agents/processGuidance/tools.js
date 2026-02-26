import { tool } from 'ai';
import { z } from 'zod';

export const getNextStepTool = tool({
    description: 'Get recommended next steps for a process (under construction)',
    parameters: z.object({
        processType: z.string().describe('Type of process or workflow'),
        currentState: z.string().optional().describe('Current state in the process'),
    }),
    execute: async ({ processType, currentState }) => {
        return {
            success: false,
            message: `Process guidance for "${processType}" is under construction. Lazy devs are working on it!`
        };
    },
});
