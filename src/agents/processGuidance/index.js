export const whatIdo = "Process Guidance Agent: Recommend next steps and provide workflow guidance.";

export async function processGuidanceAgent(query, context = {}) {
    const { emitEvent } = context;

    if (emitEvent) {
        emitEvent('step', { type: 'agent-active', content: `[Process Guidance Agent] Processing...` });
    }

    return {
        text: "🚧 **Process Guidance Agent Under Construction** 🚧\n\nOur lazy devs are mapping out the workflows! Soon I'll answer questions like 'What should I do next?' or 'What's the process for a differing site condition?'. Stay tuned!",
        steps: []
    };
}
