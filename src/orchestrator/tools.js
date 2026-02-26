import { tool } from 'ai';
import { z } from 'zod';
import {
    navigationAgent,
    dataRetrievalAgent,
    documentAnalysisAgent,
    helpAgent,
    documentGenerationAgent,
    processGuidanceAgent,
    storiesAgent,
    auditsAgent
} from '../agents/index.js';

export const getTools = (context, emitEvent, currentDate) => {
    return {
        transferToNavigationAgent: tool({
            description: 'Transfer to the Navigation Agent. Use for requests to go to specific pages or sections (e.g., "Take me to my projects", "Open the submittal register").',
            parameters: z.object({
                query: z.string().describe('The user query to pass to the navigation agent.'),
            }),
            execute: async ({ query }) => {
                emitEvent('step', { type: 'routing', agent: 'Navigation Agent', content: `Routing to Navigation Agent: "${query}"` });
                const agentResult = await navigationAgent(query, { ...context, currentDate, emitEvent });
                emitEvent('step', { type: 'tool-result', result: `Navigation Agent finished.` });
                return `Navigation Agent Response: ${agentResult.text}`;
            },
        }),

        transferToDataRetrievalAgent: tool({
            description: 'Transfer to the Data Retrieval Agent. Use for querying or fetching data about companies, opportunities, projects, solicitations, jobs, account data, or existing project documents (letters, reports, correspondence).',
            parameters: z.object({
                query: z.string().describe('The user query to pass to the data retrieval agent.'),
            }),
            execute: async ({ query }) => {
                emitEvent('step', { type: 'routing', agent: 'Data Retrieval Agent', content: `Routing to Data Retrieval Agent: "${query}"` });
                const agentResult = await dataRetrievalAgent(query, { ...context, currentDate, emitEvent });
                emitEvent('step', { type: 'tool-result', result: `Data Retrieval Agent finished.` });
                return `Data Retrieval Agent Response: ${agentResult.text}`;
            },
        }),

        transferToDocumentAnalysisAgent: tool({
            description: 'Transfer to the Document Analysis Agent. Use for reviewing submittals against specs, extracting variables from documents, or creating registers.',
            parameters: z.object({
                query: z.string().describe('The user query to pass to the document analysis agent.'),
            }),
            execute: async ({ query }) => {
                emitEvent('step', { type: 'routing', agent: 'Document Analysis Agent', content: `Routing to Document Analysis Agent: "${query}"` });
                const agentResult = await documentAnalysisAgent(query, { ...context, currentDate, emitEvent });
                emitEvent('step', { type: 'tool-result', result: `Document Analysis Agent finished.` });
                return `Document Analysis Agent Response: ${agentResult.text}`;
            },
        }),

        transferToHelpAgent: tool({
            description: 'Transfer to the Help Agent. Use for platform help, tutorials, how-to questions, or reference lookups (FAR/DFARS clauses, government regulations, trade standards, codes, and any compliance or regulatory question).',
            parameters: z.object({
                query: z.string().describe('The user query to pass to the help agent.'),
            }),
            execute: async ({ query }) => {
                emitEvent('step', { type: 'routing', agent: 'Help Agent', content: `Routing to Help Agent: "${query}"` });
                const agentResult = await helpAgent(query, { ...context, currentDate, emitEvent });
                console.log('[Orchestrator] Help Agent response:', agentResult.text?.substring(0, 500));
                emitEvent('step', { type: 'tool-result', result: `Help Agent finished.` });
                return `Help Agent Response: ${agentResult.text}`;
            },
        }),

        transferToDocumentGenerationAgent: tool({
            description: 'Transfer to the Document Generation Agent. Use for creating NEW documents like serial letters, REAs, monthly narratives (not for fetching existing ones).',
            parameters: z.object({
                query: z.string().describe('The user query to pass to the document generation agent.'),
            }),
            execute: async ({ query }) => {
                emitEvent('step', { type: 'routing', agent: 'Document Generation Agent', content: `Routing to Document Generation Agent: "${query}"` });
                const agentResult = await documentGenerationAgent(query, { ...context, currentDate, emitEvent });
                emitEvent('step', { type: 'tool-result', result: `Document Generation Agent finished.` });
                return `Document Generation Agent Response: ${agentResult.text}`;
            },
        }),

        transferToProcessGuidanceAgent: tool({
            description: 'Transfer to the Process Guidance Agent. Use for "what should I do next?" questions, workflow guidance, or process recommendations.',
            parameters: z.object({
                query: z.string().describe('The user query to pass to the process guidance agent.'),
            }),
            execute: async ({ query }) => {
                emitEvent('step', { type: 'routing', agent: 'Process Guidance Agent', content: `Routing to Process Guidance Agent: "${query}"` });
                const agentResult = await processGuidanceAgent(query, { ...context, currentDate, emitEvent });
                emitEvent('step', { type: 'tool-result', result: `Process Guidance Agent finished.` });
                return `Process Guidance Agent Response: ${agentResult.text}`;
            },
        }),

        transferToStoriesAgent: tool({
            description: 'Transfer to the Stories Agent. Use for creating document collections, managing REA stories, or gathering project documentation.',
            parameters: z.object({
                query: z.string().describe('The user query to pass to the stories agent.'),
            }),
            execute: async ({ query }) => {
                emitEvent('step', { type: 'routing', agent: 'Stories Agent', content: `Routing to Stories Agent: "${query}"` });
                const agentResult = await storiesAgent(query, { ...context, currentDate, emitEvent });
                emitEvent('step', { type: 'tool-result', result: `Stories Agent finished.` });
                return `Stories Agent Response: ${agentResult.text}`;
            },
        }),

        transferToAuditsAgent: tool({
            description: 'Transfer to the Audits Agent. Use for project monitoring, schedule performance queries, delay alerts, or proactive insights.',
            parameters: z.object({
                query: z.string().describe('The user query to pass to the audits agent.'),
            }),
            execute: async ({ query }) => {
                emitEvent('step', { type: 'routing', agent: 'Audits Agent', content: `Routing to Audits Agent: "${query}"` });
                const agentResult = await auditsAgent(query, { ...context, currentDate, emitEvent });
                emitEvent('step', { type: 'tool-result', result: `Audits Agent finished.` });
                return `Audits Agent Response: ${agentResult.text}`;
            },
        }),
    };
};
