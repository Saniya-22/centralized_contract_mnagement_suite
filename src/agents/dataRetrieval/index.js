export const whatIdo = "Data Retrieval Agent: Query companies, opportunities, projects, solicitations, and account data.";

export async function dataRetrievalAgent(query, context = {}) {
    const { emitEvent } = context;

    if (emitEvent) {
        emitEvent('step', { type: 'agent-active', content: `[Data Retrieval Agent] Processing...` });
    }

    return {
        text: "🚧 **Data Retrieval Agent Under Construction** 🚧\n\nOur lazy devs are still wiring up the data pipes! Soon you'll be able to ask things like 'Find DBE certified contractors in Seattle' or 'What opportunities are closing this week?'. Stay tuned!",
        steps: []
    };
}
