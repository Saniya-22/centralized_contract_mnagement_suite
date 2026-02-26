import { streamText, generateText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { getSystemPrompt } from './prompts.js';
import { helpAgentTools } from './tools.js';

export const whatIdo = "Help Agent: Access government regulations (FAR, DFARS, EM 385-1-1), provide contextual help, and explain contracting requirements.";

export async function helpAgent(query, context = {}) {
    const { emitEvent, currentDate } = context;

    if (emitEvent) {
        emitEvent('step', {
            type: 'agent-active',
            content: `[Help Agent] Searching regulations and generating response...`
        });
    }

    const systemPrompt = getSystemPrompt({ currentDate: currentDate || new Date().toISOString().split('T')[0] });

    const steps = [];

    const wrappedTools = {};
    for (const [toolName, toolDef] of Object.entries(helpAgentTools)) {
        wrappedTools[toolName] = {
            ...toolDef,
            execute: async (params) => {
                if (emitEvent) {
                    emitEvent('step', {
                        type: 'tool-call',
                        tool: toolName,
                        args: JSON.stringify(params),
                        content: `Calling ${toolName}: ${JSON.stringify(params)}`
                    });
                }

                const result = await toolDef.execute(params);
                console.log(">>",result)
                steps.push({
                    tool: toolName,
                    params,
                    result: result.success ? 'success' : 'failed'
                });

                if (emitEvent) {
                    emitEvent('step', {
                        type: 'tool-result',
                        tool: toolName,
                        content: result.success
                            ? `${toolName} returned ${result.returned || 'results'}`
                            : `${toolName} failed: ${result.message}`
                    });
                }

                return result;
            }
        };
    }

    try {
        const result = await generateText({
            model: openai(process.env.MODEL_NAME || 'gpt-4o'),
            system: systemPrompt,
            messages: [
                {
                    role: 'user',
                    content: query
                }
            ],
            tools: wrappedTools,
            maxSteps: 5,
            toolChoice: 'auto'
        });

        let responseText = result.text || '';

        if (!responseText && result.steps && result.steps.length > 0) {
            const lastStep = result.steps[result.steps.length - 1];
            if (lastStep.toolResults) {
                responseText = formatToolResults(lastStep.toolResults);
            }
        }

        if (responseText && !responseText.includes('Disclaimer')) {
            responseText += '\n\n> **Disclaimer**: This information is for general reference purposes only. Government regulations are subject to change. For contract-specific decisions or legal matters, consult official source documents and/or legal counsel.';
        }

        if (emitEvent) {
            emitEvent('step', {
                type: 'agent-complete',
                content: `[Help Agent] Response generated successfully.`
            });
        }

        return {
            text: responseText || "I couldn't find relevant regulation information for your query. Please try rephrasing your question or specifying the regulation type (FAR, DFARS, or EM 385).",
            steps
        };

    } catch (error) {
        console.error('Help Agent error:', error);

        if (emitEvent) {
            emitEvent('step', {
                type: 'agent-error',
                content: `[Help Agent] Error: ${error.message}`
            });
        }

        return {
            text: `I encountered an issue while searching the regulations database. Please try again or rephrase your question.\n\n**Error Details**: ${error.message}\n\n> If this persists, the regulation database may be temporarily unavailable. You can also search the official sources:\n> - [FAR](https://www.acquisition.gov/far/)\n> - [DFARS](https://www.acquisition.gov/dfars/)`,
            steps
        };
    }
}

function formatToolResults(toolResults) {
    const parts = [];

    for (const result of toolResults) {
        if (result.result) {
            const data = result.result;

            if (data.success && data.context) {
                parts.push(data.context);
            } else if (data.success && data.clause) {
                parts.push(`**${data.clauseReference}**\n\n${data.clause.text}`);
            } else if (data.message) {
                parts.push(data.message);
            }
        }
    }

    return parts.join('\n\n---\n\n');
}

export async function helpAgentStreaming(query, context = {}) {
    const { emitEvent, currentDate } = context;

    if (emitEvent) {
        emitEvent('step', {
            type: 'agent-active',
            content: `[Help Agent] Processing...`
        });
    }

    const systemPrompt = getSystemPrompt({ currentDate: currentDate || new Date().toISOString().split('T')[0] });

    const result = await streamText({
        model: openai(process.env.MODEL_NAME || 'gpt-4o'),
        system: systemPrompt,
        messages: [{ role: 'user', content: query }],
        tools: helpAgentTools,
        maxSteps: 5,
        toolChoice: 'auto',
        onStepFinish: (step) => {
            if (emitEvent && step.toolCalls) {
                for (const call of step.toolCalls) {
                    emitEvent('step', {
                        type: 'tool-call',
                        tool: call.toolName,
                        content: `Called ${call.toolName}`
                    });
                }
            }
        }
    });

    return result;
}
