import { provisionUser, login as authLogin } from '../../auth/auth.service.js';
import { signToken } from '../../auth/jwt.js';

const ADMIN_API_KEY = process.env.ADMIN_API_KEY;

/**
 * POST /login
 * Body: { email, password }
 * Returns: { token, user } or 401 with lock info / invalid credentials.
 */
async function loginRoute(request, reply) {
    const { email, password } = request.body || {};
    const result = await authLogin(email, password, request.log);

    if (!result.success) {
        if (result.code === 'ACCOUNT_LOCKED' || result.accountLocked) {
            return reply.code(423).send({
                error: 'Account locked',
                message: 'Too many failed login attempts. Account is temporarily locked.',
                lockUntil: result.lockUntil,
                remainingMinutes: result.remainingMinutes
            });
        }
        if (result.code === 'INVALID_INPUT') {
            return reply.code(400).send({
                error: 'Bad request',
                message: 'Email and password are required.'
            });
        }
        return reply.code(401).send({
            error: 'Invalid credentials',
            message: 'Invalid email or password.'
        });
    }

    const token = signToken(result.user);
    // Set HttpOnly cookie for browser clients (optional). Cookie max age defaults to 7 days.
    const cookieMaxAge = process.env.COOKIE_MAX_AGE ? parseInt(process.env.COOKIE_MAX_AGE, 10) : 7 * 24 * 60 * 60;
    try {
        reply.setCookie && reply.setCookie('token', token, {
            httpOnly: true,
            path: '/',
            sameSite: 'lax',
            secure: process.env.NODE_ENV === 'production',
            maxAge: cookieMaxAge
        });
    } catch (err) {
        // If cookie plugin isn't registered, ignore and continue
        request.log.warn('Failed to set auth cookie (cookie plugin may be missing).');
    }

    return reply.send({
        token,
        user: result.user,
        expiresIn: process.env.JWT_EXPIRES_IN || '7d'
    });
}

/**
 * POST /admin/users (account provisioning)
 * Header: X-Admin-Key: <ADMIN_API_KEY> (or Authorization: Bearer <ADMIN_API_KEY>)
 * Body: { fullName, email }
 * Returns: { user, tempPasswordSent }. Temp password is stubbed (logged only).
 */
async function provisionUserRoute(request, reply) {
    if (!ADMIN_API_KEY) {
        return reply.code(503).send({
            error: 'Service unavailable',
            message: 'Admin provisioning is not configured (ADMIN_API_KEY missing).'
        });
    }

    const adminKey = request.headers['x-admin-key'] ?? request.headers.authorization?.replace(/^Bearer\s+/i, '');
    if (adminKey !== ADMIN_API_KEY) {
        return reply.code(403).send({
            error: 'Forbidden',
            message: 'Invalid or missing admin key.'
        });
    }

    const { fullName, email } = request.body || {};
    if (!fullName?.trim() || !email?.trim()) {
        return reply.code(400).send({
            error: 'Bad request',
            message: 'fullName and email are required.'
        });
    }

    try {
        const result = await provisionUser({ fullName: fullName.trim(), email: email.trim() }, request.log);
        return reply.code(201).send(result);
    } catch (err) {
        if (err.code === '23505') {
            return reply.code(409).send({
                error: 'Conflict',
                message: 'A user with this email already exists.'
            });
        }
        request.log.error(err);
        return reply.code(500).send({
            error: 'Internal server error',
            message: 'Failed to provision user.'
        });
    }
}

export default async function authRoutes(fastify) {
    fastify.post('/login', {
        schema: {
            body: {
                type: 'object',
                required: ['email', 'password'],
                properties: {
                    email: { type: 'string' },
                    password: { type: 'string' }
                }
            }
        }
    }, loginRoute);

    fastify.post('/admin/users', {
        schema: {
            body: {
                type: 'object',
                required: ['fullName', 'email'],
                properties: {
                    fullName: { type: 'string' },
                    email: { type: 'string', format: 'email' }
                }
            }
        }
    }, provisionUserRoute);

    // POST /logout - clears auth cookie (for browser clients)
    fastify.post('/logout', async function (request, reply) {
        try {
            // Clear the token cookie if cookie plugin is available
            try {
                reply.clearCookie && reply.clearCookie('token', { path: '/' });
            } catch (e) {
                request.log.warn('Failed to clear cookie during logout', e);
            }
            return reply.send({ success: true, message: 'Logged out' });
        } catch (err) {
            request.log.error(err);
            return reply.code(500).send({ error: 'Internal server error' });
        }
    });
}
