/**
 * Migration script to create the chat_history table.
 *
 * Idempotent — safe to run multiple times (uses IF NOT EXISTS).
 *
 * Usage: yarn migrate:chatHistory
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
        console.log('[ChatHistory Migration] Starting...');

        // 1. Create table
        console.log('[ChatHistory Migration] Creating chat_history table...');
        await client.query(`
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255),
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        `);

        // 2. Create indexes
        console.log('[ChatHistory Migration] Creating indexes...');
        await client.query(`
            CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history (session_id, created_at);
        `);

        // 3. Verify
        const result = await client.query(`
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'chat_history'
            ORDER BY ordinal_position;
        `);

        console.log('[ChatHistory Migration] Table columns:');
        result.rows.forEach(row => {
            console.log(`  ${row.column_name} (${row.data_type})`);
        });

        console.log('[ChatHistory Migration] Done!');
    } catch (error) {
        console.error('[ChatHistory Migration] Error:', error.message);
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
