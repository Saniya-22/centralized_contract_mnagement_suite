import { tool } from 'ai';
import { z } from 'zod';

export const auditProjectTool = tool({
    description: 'Run project audits and check for delays (under construction)',
    parameters: z.object({
        projectId: z.string().describe('Project to audit'),
        auditType: z.enum(['schedule', 'submittals', 'rfi', 'resources']).optional().describe('Type of audit'),
    }),
    execute: async ({ projectId, auditType }) => {
        return {
            success: false,
            message: `Project audit for "${projectId}" is under construction. Lazy devs are working on it!`
        };
    },
});
