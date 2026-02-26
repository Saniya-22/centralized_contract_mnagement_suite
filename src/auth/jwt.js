import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET;
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';

/**
 * Issue a JWT for a user (after successful login).
 */
export function signToken(payload) {
    if (!JWT_SECRET) throw new Error('JWT_SECRET is required to sign tokens');
    return jwt.sign(
        {
            id: payload.id,
            email: payload.email,
            fullName: payload.fullName
        },
        JWT_SECRET,
        { expiresIn: JWT_EXPIRES_IN }
    );
}
