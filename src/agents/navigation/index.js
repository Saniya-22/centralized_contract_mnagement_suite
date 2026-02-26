export const whatIdo = "Navigation Agent: Route users to pages via natural language commands.";

export async function navigationAgent(query, context = {}) {
    const { emitEvent } = context;

    if (emitEvent) {
        emitEvent('step', { type: 'agent-active', content: `[Navigation Agent] Processing...` });
    }

    return {
        text: "🚧 **Navigation Agent Under Construction** 🚧\n\nOur lazy devs haven't finished building this yet! Soon you'll be able to navigate anywhere in GovGig using natural language like 'Take me to my project submittals' or 'Open opportunity listings'. Check back soon!",
        steps: []
    };
}
