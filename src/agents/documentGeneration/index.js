export const whatIdo = "Document Generation Agent: Generate serial letters, REAs, and monthly narratives from project data.";

export async function documentGenerationAgent(query, context = {}) {
    const { emitEvent } = context;

    if (emitEvent) {
        emitEvent('step', { type: 'agent-active', content: `[Document Generation Agent] Processing...` });
    }

    return {
        text: "🚧 **Document Generation Agent Under Construction** 🚧\n\nOur lazy devs are still training me to write like a project manager! Soon I'll generate serial letters from RFIs, draft REAs, and create monthly narratives. Coming soon!",
        steps: []
    };
}
