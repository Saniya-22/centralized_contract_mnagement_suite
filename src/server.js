import 'dotenv/config';
import Fastify from 'fastify';
import cors from '@fastify/cors';
import chatRoutes from './routes/chat/index.js';
import authRoutes from './routes/auth/index.js';
import fastifyCookie from '@fastify/cookie';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { requireAuthOrRedirect } from './auth/auth.middleware.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const fastify = Fastify({ logger: true });

fastify.register(cors, { origin: '*' });
fastify.register(fastifyCookie, { secret: process.env.COOKIE_SECRET || 'dev_cookie_secret' });

fastify.get('/health', async () => ({ status: 'ok' }));

// Auth: login and admin provisioning (no auth required on these)
fastify.register(authRoutes, { prefix: '/cai/api/auth' });
// Serve login page (create `public/login.html` if missing)
fastify.get('/login', async (request, reply) => {
    const loginPath = path.join(__dirname, '../public/login.html');
    try {
        if (fs.existsSync(loginPath)) {
            const html = fs.readFileSync(loginPath, 'utf8');
            return reply.type('text/html').send(html);
        }
        // Fallback JSON if file missing
        return reply.code(200).send({
            message: 'Use POST /cai/api/auth/login with body { email, password } to authenticate.',
            loginEndpoint: '/cai/api/auth/login'
        });
    } catch (err) {
        request.log.error('Failed to serve /login page', err);
        return reply.code(500).send({ error: 'Internal server error' });
    }
});

fastify.register(chatRoutes, { prefix: '/cai/api/chat' });

// Serve /stream only for authenticated users (redirect to /login if not authenticated)
fastify.get('/stream', { preHandler: requireAuthOrRedirect }, async (request, reply) => {
    const streamHtml = fs.readFileSync(path.join(__dirname, '../public/stream.html'), 'utf8');
    reply.type('text/html').send(streamHtml);
});

fastify.get('/feedback', async (request, reply) => {
    const feedbackHtml = fs.readFileSync(path.join(__dirname, '../public/feedback.html'), 'utf8');
    reply.type('text/html').send(feedbackHtml);
});

const start = async () => {
    try {
        const port = process.env.PORT || 8000;
        await fastify.listen({ port, host: '0.0.0.0' });
        console.log(`Server running at http://localhost:${port}. Open http://localhost:${port}/stream to test.`);
    } catch (err) {
        fastify.log.error(err);
        process.exit(1);
    }
};

start();
