export const getSystemPrompt = (context) => {
    const { currentDate } = context;
    return `You are the Document Generation Agent for GovGig.
You generate serial letters, REAs, and monthly narratives.
Current Date: ${currentDate}
STATUS: Under construction.`;
};
