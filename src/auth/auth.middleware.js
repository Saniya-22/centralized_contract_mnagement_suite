import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET;

/**
 * Require a valid JWT for the request. Use as preHandler on protected routes.
 * Expects: Authorization: Bearer <token> or cookie/session (optional).
 * On success: request.user = { id, email, fullName }.
 * On failure: reply 401 with { error, message }.
 */
export function requireAuth(request, reply, done) {
    if (!JWT_SECRET) {
        return reply.code(500).send({
            error: 'Unauthorized',
            message: 'Authentication not configured (JWT_SECRET missing).'
        });
    }

    const authHeader = request.headers.authorization;
    const token = authHeader?.startsWith('Bearer ')
        ? authHeader.slice(7)
        : request.cookies?.token ?? null;

    if (!token) {
        return reply.code(401).send({
            error: 'Unauthorized',
            message: 'Authentication required. Provide Authorization: Bearer <token> or log in at /login.'
        });
    }

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        request.user = {
            id: decoded.id,
            email: decoded.email,
            fullName: decoded.fullName
        };
        done();
    } catch (err) {
        return reply.code(401).send({
            error: 'Unauthorized',
            message: 'Invalid or expired token.'
        });
    }
}

/**
 * Require authentication for pages (HTML) and redirect to /login when absent or invalid.
 * Use as preHandler for routes serving HTML (e.g. /stream).
 */
export function requireAuthOrRedirect(request, reply, done) {
    if (!JWT_SECRET) {
        // If auth not configured, redirect to login page
        return reply.redirect('/login');
    }

    const authHeader = request.headers.authorization;
    const token = authHeader?.startsWith('Bearer ')
        ? authHeader.slice(7)
        : request.cookies?.token ?? null;

    if (!token) {
        return reply.redirect('/login');
    }

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        request.user = {
            id: decoded.id,
            email: decoded.email,
            fullName: decoded.fullName
        };
        done();
    } catch (err) {
        return reply.redirect('/login');
    }
}
