import { openai } from '@ai-sdk/openai';
import { streamText, tool } from 'ai';
import { z } from 'zod';
import { getSystemPrompt } from './prompts.js';
import { getTools } from './tools.js';

const MODEL_NAME = process.env.MODEL_NAME || "gpt-4o";

export async function runOrchestrator(query, context = {}, emitEvent) {
    const { person_id, history = [], cot = false } = context;

    const now = new Date();
    const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
    const currentDate = now.toLocaleDateString('en-US', options);

    const log = (msg) => emitEvent('step', { type: 'agent-active', content: msg });

    log(`Orchestrator received query: "${query}"`);

    if (cot) {
        log(`[CoT Mode] Active. Using 'think' tool for reasoning.`);
    }

    let tools = getTools(context, emitEvent, currentDate);

    if (cot) {
        tools = {
            ...tools,
            think: tool({
                description: 'Record your step-by-step reasoning plan. Call this BEFORE any other tools or answer.',
                parameters: z.object({
                    thought: z.string().describe('The thinking process text, explaining your plan.'),
                }),
                execute: async ({ thought }) => {
                    return "Thought recorded. Proceed.";
                }
            })
        };
    }

    const wrappedTools = {};
    for (const [key, toolDef] of Object.entries(tools)) {
        wrappedTools[key] = {
            ...toolDef,
            execute: async (args, toolContext) => {
                if (key === 'think') {
                    emitEvent('step', {
                        type: 'thought',
                        content: args.thought
                    });
                    return toolDef.execute(args, toolContext);
                } else {
                    emitEvent('step', {
                        type: 'tool-call',
                        tool: key,
                        args: JSON.stringify(args)
                    });
                    return toolDef.execute(args, toolContext);
                }
            }
        };
    }

    const messages = [
        ...history,
        { role: 'user', content: query }
    ];

    try {
        if (!MODEL_NAME) {
            throw new Error('MODEL_NAME is not set');
        }
        const result = await streamText({
            model: openai(MODEL_NAME),
            system: getSystemPrompt({ person_id, currentDate, cot }),
            temperature : 0.2,
            messages: messages,
            tools: wrappedTools,
            maxSteps: 10,
            toolChoice: cot ? 'auto' : 'auto',
            onStepFinish: ({ toolCalls, toolResults }) => {
            }
        });

        for await (const delta of result.textStream) {
            emitEvent('step', { type: 'chunk', content: delta });
        }

        emitEvent('done', '[DONE]');

        return;

    } catch (error) {
        console.error("Orchestrator Error:", error);
        emitEvent('step', { type: 'thought', content: `Error: ${error.message} ` });
        emitEvent('done', '[ERROR]');
        throw error;
    }
}
