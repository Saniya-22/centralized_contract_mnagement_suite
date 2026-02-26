import bcrypt from 'bcrypt';
import crypto from 'crypto';
import { pool } from '../db/pool.js';

const BCRYPT_ROUNDS = 12;
const LOCKOUT_THRESHOLD = 5;
const LOCKOUT_DURATION_MS = 15 * 60 * 1000; // 15 minutes

/**
 * Generate a secure temporary password (12+ chars: upper, lower, digit, symbol).
 */
export function generateSecurePassword(length = 14) {
    const upper = 'ABCDEFGHJKLMNPQRSTUVWXYZ';
    const lower = 'abcdefghjkmnpqrstuvwxyz';
    const digits = '23456789';
    const symbols = '!@#$%&*';
    const all = upper + lower + digits + symbols;
    const getRandom = (source) => source[crypto.randomInt(0, source.length)];
    let password = getRandom(upper) + getRandom(lower) + getRandom(digits) + getRandom(symbols);
    for (let i = password.length; i < length; i++) {
        password += getRandom(all);
    }
    return password.split('').sort(() => crypto.randomInt(0, 2) - 1).join('');
}

/**
 * Hash password with bcrypt.
 */
export async function hashPassword(plainPassword) {
    return bcrypt.hash(plainPassword, BCRYPT_ROUNDS);
}

/**
 * Compare plain password with hashed password.
 */
export async function comparePassword(plainPassword, hashedPassword) {
    return bcrypt.compare(plainPassword, hashedPassword);
}

/**
 * Stub: simulate secure delivery of temporary password (log; email stub).
 */
export function deliverTempPassword(email, fullName, tempPassword, log) {
    const logger = log || console;
    logger.info({
        event: 'temp_password_delivery',
        email,
        fullName,
        message: 'Temporary password generated. In production, send via secure channel (e.g. email).'
    });
    // Stub for email: in production call your email service here
    // await emailService.sendTempPassword({ to: email, fullName, tempPassword });
}

/**
 * Provision a new user (admin/system). Creates account with hashed temp password.
 */
export async function provisionUser({ fullName, email }, log) {
    const tempPassword = generateSecurePassword(14);
    const hashedPassword = await hashPassword(tempPassword);

    const result = await pool.query(
        `INSERT INTO users (full_name, email, hashed_password, is_locked, failed_login_attempts)
         VALUES ($1, $2, $3, FALSE, 0)
         RETURNING id, full_name, email, created_at`,
        [fullName, email.trim().toLowerCase(), hashedPassword]
    );

    const user = result.rows[0];
    deliverTempPassword(email, fullName, tempPassword, log);
    return {
        user: {
            id: user.id,
            fullName: user.full_name,
            email: user.email,
            created_at: user.created_at
        },
        tempPassword,
        message: 'Share this temporary password with the user securely. It is only returned once.'
    };
}

/**
 * Log lockout event to auth_audit_log.
 */
export async function logLockoutEvent(userId, email, details, log) {
    const logger = log || console;
    logger.warn({ event: 'account_lockout', userId, email, ...details });
    await pool.query(
        `INSERT INTO auth_audit_log (user_id, email, event_type, details)
         VALUES ($1, $2, 'lockout', $3)`,
        [userId, email, JSON.stringify(details)]
    );
}

/**
 * Clear lock and reset failed attempts when lock period has expired.
 */
async function clearExpiredLock(user) {
    if (!user.is_locked || !user.lock_until) return user;
    const now = new Date();
    if (new Date(user.lock_until) <= now) {
        await pool.query(
            `UPDATE users SET is_locked = FALSE, lock_until = NULL, failed_login_attempts = 0, updated_at = $1 WHERE id = $2`,
            [now, user.id]
        );
        return { ...user, is_locked: false, lock_until: null, failed_login_attempts: 0 };
    }
    return user;
}

/**
 * Record failed login and optionally lock account.
 */
async function recordFailedLogin(user, log) {
    const attempts = (user.failed_login_attempts || 0) + 1;
    const now = new Date();
    const lockUntil = attempts >= LOCKOUT_THRESHOLD ? new Date(now.getTime() + LOCKOUT_DURATION_MS) : null;
    const isLocked = attempts >= LOCKOUT_THRESHOLD;

    await pool.query(
        `UPDATE users SET failed_login_attempts = $1, is_locked = $2, lock_until = $3, updated_at = $4 WHERE id = $5`,
        [attempts, isLocked, lockUntil, now, user.id]
    );

    if (isLocked) {
        await logLockoutEvent(
            user.id,
            user.email,
            { failedAttempts: attempts, lockUntil: lockUntil?.toISOString(), lockDurationMinutes: 15 },
            log
        );
    }

    return {
        locked: isLocked,
        failedAttempts: attempts,
        lockUntil: lockUntil?.toISOString() ?? null,
        remainingMs: lockUntil ? lockUntil - now : 0
    };
}

/**
 * Authenticate user by email (username) and password. Returns user + token payload data or lock info.
 */
export async function login(email, plainPassword, log) {
    const logger = log || console;
    const normalizedEmail = email?.trim()?.toLowerCase();
    if (!normalizedEmail || !plainPassword) {
        return { success: false, code: 'INVALID_INPUT' };
    }

    const userResult = await pool.query(
        `SELECT id, full_name, email, hashed_password, is_locked, lock_until, failed_login_attempts FROM users WHERE email = $1`,
        [normalizedEmail]
    );

    if (userResult.rows.length === 0) {
        return { success: false, code: 'INVALID_CREDENTIALS' };
    }

    const user = userResult.rows[0];
    const u = await clearExpiredLock({
        ...user,
        is_locked: user.is_locked,
        lock_until: user.lock_until,
        failed_login_attempts: user.failed_login_attempts
    });

    if (u.is_locked && u.lock_until) {
        const now = new Date();
        const until = new Date(u.lock_until);
        if (until > now) {
            const remainingMs = until - now;
            return {
                success: false,
                code: 'ACCOUNT_LOCKED',
                lockUntil: until.toISOString(),
                remainingMinutes: Math.ceil(remainingMs / 60000)
            };
        }
    }

    const passwordMatch = await comparePassword(plainPassword, user.hashed_password);
    if (!passwordMatch) {
        const updated = await recordFailedLogin(u, logger);
        return {
            success: false,
            code: 'INVALID_CREDENTIALS',
            ...(updated.locked && {
                accountLocked: true,
                lockUntil: updated.lockUntil,
                remainingMinutes: Math.ceil(updated.remainingMs / 60000)
            })
        };
    }

    // Success: reset failed attempts
    await pool.query(
        `UPDATE users SET failed_login_attempts = 0, is_locked = FALSE, lock_until = NULL, updated_at = $1 WHERE id = $2`,
        [new Date(), user.id]
    );

    return {
        success: true,
        user: {
            id: user.id,
            fullName: user.full_name,
            email: user.email
        }
    };
}
