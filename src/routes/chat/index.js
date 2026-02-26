import { runOrchestrator } from '../../orchestrator/index.js';
import { requireAuth } from '../../auth/auth.middleware.js';
import { pool } from '../../db/pool.js';

export default async function (fastify, opts) {
    fastify.get('/bot_response', { preHandler: requireAuth }, async function (request, reply) {
        const { query, session_id, cot } = request.query;
        const { id: userId } = request.user;

        console.log(`[API] Received query: ${query} (CoT: ${cot})`);

        if (!query) {
            return reply.code(400).send({ status: 'error', message: 'Query parameter is required' });
        }

        const currentSessionId = session_id || 'default-session';

        reply.raw.setHeader('Content-Type', 'text/event-stream');
        reply.raw.setHeader('Cache-Control', 'no-cache');
        reply.raw.setHeader('Connection', 'keep-alive');

        const emitEvent = (event, data) => {
            if (reply.raw.destroyed) return;
            const dataStr = typeof data === 'string' ? data : JSON.stringify(data);
            reply.raw.write(`event: ${event}\ndata: ${dataStr}\n\n`);
        };

        try {
            // Fetch last 5 exchanges (10 rows) for this session
            const historyResult = await pool.query(
                `SELECT role, content FROM chat_history
                 WHERE session_id = $1
                 ORDER BY created_at DESC LIMIT 10`,
                [currentSessionId]
            );
            const history = historyResult.rows.reverse();

            // Save user message
            await pool.query(
                `INSERT INTO chat_history (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)`,
                [currentSessionId, userId, 'user', query]
            );

            // Wrap emitEvent to accumulate the full response
            let fullResponse = '';
            const wrappedEmitEvent = (event, data) => {
                if (event === 'step' && typeof data === 'object' && data.type === 'chunk') {
                    fullResponse += data.content;
                }
                emitEvent(event, data);
            };

            await runOrchestrator(query, {
                person_id: userId,
                chat_id: currentSessionId,
                history,
                cot: cot === 'true'
            }, wrappedEmitEvent);

            // Save assistant message after stream completes
            if (fullResponse) {
                await pool.query(
                    `INSERT INTO chat_history (session_id, user_id, role, content) VALUES ($1, $2, $3, $4)`,
                    [currentSessionId, userId, 'assistant', fullResponse]
                );
            }

            if (!reply.raw.destroyed) {
                reply.raw.end();
            }

        } catch (error) {
            console.error(error);
            emitEvent('error', { message: 'Internal Server Error' });
            if (!reply.raw.destroyed) {
                reply.raw.end();
            }
        }
    });

    fastify.get('/feedback', async function (request, reply) {
        const { rating, email, limit = 50, offset = 0 } = request.query;

        try {
            let sql = `SELECT f.id, f.rating, f.message, (f.screenshot IS NOT NULL) AS has_screenshot, f.query, f.response, f.agent_name, f.session_id, f.user_id, u.email AS user_email, f.created_at FROM chat_feedback f LEFT JOIN users u ON f.user_id::integer = u.id`;
            const params = [];
            const conditions = [];

            if (rating && ['up', 'down'].includes(rating)) {
                params.push(rating);
                conditions.push(`f.rating = $${params.length}`);
            }

            if (email && email.trim()) {
                params.push(`%${email.trim().toLowerCase()}%`);
                conditions.push(`LOWER(u.email) LIKE $${params.length}`);
            }

            if (conditions.length > 0) {
                sql += ` WHERE ${conditions.join(' AND ')}`;
            }

            sql += ' ORDER BY f.created_at DESC';
            params.push(parseInt(limit));
            sql += ` LIMIT $${params.length}`;
            params.push(parseInt(offset));
            sql += ` OFFSET $${params.length}`;

            const result = await pool.query(sql, params);
            return reply.send(result.rows);
        } catch (error) {
            console.error('[Feedback] Query error:', error);
            return reply.code(500).send({ status: 'error', message: 'Failed to fetch feedback' });
        }
    });

    fastify.get('/feedback/:id/screenshot', async function (request, reply) {
        const { id } = request.params;

        try {
            const result = await pool.query('SELECT screenshot FROM chat_feedback WHERE id = $1', [id]);
            if (result.rows.length === 0 || !result.rows[0].screenshot) {
                return reply.code(404).send({ status: 'error', message: 'Screenshot not found' });
            }
            return reply.send({ screenshot: result.rows[0].screenshot });
        } catch (error) {
            console.error('[Feedback] Screenshot query error:', error);
            return reply.code(500).send({ status: 'error', message: 'Failed to fetch screenshot' });
        }
    });

    fastify.delete('/feedback/:id', async function (request, reply) {
        const { id } = request.params;

        try {
            const result = await pool.query('DELETE FROM chat_feedback WHERE id = $1', [id]);
            if (result.rowCount === 0) {
                return reply.code(404).send({ status: 'error', message: 'Feedback not found' });
            }
            return reply.send({ status: 'ok' });
        } catch (error) {
            console.error('[Feedback] Delete error:', error);
            return reply.code(500).send({ status: 'error', message: 'Failed to delete feedback' });
        }
    });

    fastify.post('/feedback', { preHandler: requireAuth, bodyLimit: 10 * 1024 * 1024 }, async function (request, reply) {
        const { rating, message, screenshot, query, response, agent_name, session_id } = request.body || {};
        const user_id = request.user.id;

        if (!rating || !['up', 'down'].includes(rating)) {
            return reply.code(400).send({ status: 'error', message: 'rating must be "up" or "down"' });
        }

        if (rating === 'down' && (!message || !message.trim())) {
            return reply.code(400).send({ status: 'error', message: 'Message is required for negative feedback' });
        }

        try {
            await pool.query(
                `INSERT INTO chat_feedback (rating, message, screenshot, query, response, agent_name, session_id, user_id)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8)`,
                [rating, message || null, screenshot || null, query || null, response || null, agent_name || null, session_id || null, user_id || null]
            );
            return reply.send({ status: 'ok' });
        } catch (error) {
            console.error('[Feedback] Insert error:', error);
            return reply.code(500).send({ status: 'error', message: 'Failed to save feedback' });
        }
    });
};
