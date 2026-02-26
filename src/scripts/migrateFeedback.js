/**
 * Migration script to create the chat_feedback table.
 *
 * Idempotent — safe to run multiple times (uses IF NOT EXISTS).
 *
 * Usage: yarn migrate:feedback
 */

import pg from 'pg';
import dotenv from 'dotenv';

dotenv.config();

const {
    PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS, PG_SSLMODE
} = process.env;

const ssl = PG_SSLMODE === 'require' ? { rejectUnauthorized: false } : false;

async function migrate() {
    const pool = new pg.Pool({
        host: PG_HOST,
        port: parseInt(PG_PORT || '5432'),
        database: PG_DB,
        user: PG_USER,
        password: PG_PASS,
        ssl,
    });

    const client = await pool.connect();

    try {
        console.log('[Feedback Migration] Starting...');

        // 1. Create table
        console.log('[Feedback Migration] Creating chat_feedback table...');
        await client.query(`
            CREATE TABLE IF NOT EXISTS chat_feedback (
                id SERIAL PRIMARY KEY,
                rating VARCHAR(10) NOT NULL,
                message TEXT,
                screenshot TEXT,
                query TEXT,
                response TEXT,
                agent_name VARCHAR(100),
                session_id VARCHAR(255),
                user_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT NOW()
            );
        `);

        // 2. Create indexes
        console.log('[Feedback Migration] Creating indexes...');
        await client.query(`
            CREATE INDEX IF NOT EXISTS idx_chat_feedback_rating ON chat_feedback (rating);
        `);
        await client.query(`
            CREATE INDEX IF NOT EXISTS idx_chat_feedback_created_at ON chat_feedback (created_at);
        `);

        // 3. Verify
        const result = await client.query(`
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'chat_feedback'
            ORDER BY ordinal_position;
        `);

        console.log('[Feedback Migration] Table columns:');
        result.rows.forEach(row => {
            console.log(`  ${row.column_name} (${row.data_type})`);
        });

        console.log('[Feedback Migration] Done!');
    } catch (error) {
        console.error('[Feedback Migration] Error:', error.message);
        throw error;
    } finally {
        client.release();
        await pool.end();
    }
}

migrate().catch(err => {
    console.error(err);
    process.exit(1);
});
