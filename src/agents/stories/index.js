export const whatIdo = "Stories Agent: Create and manage document collections for claims, REAs, and project documentation.";

export async function storiesAgent(query, context = {}) {
    const { emitEvent } = context;

    if (emitEvent) {
        emitEvent('step', { type: 'agent-active', content: `[Stories Agent] Processing...` });
    }

    return {
        text: "🚧 **Stories Agent Under Construction** 🚧\n\nOur lazy devs are building the story collection system! Soon you'll be able to create document collections like 'Collect all documents related to the foundation delay' or manage REA stories. Check back soon!",
        steps: []
    };
}
