#!/usr/bin/env node
import 'dotenv/config';
import pg from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const {
    PG_HOST = 'localhost',
    PG_PORT = '5432',
    PG_DB = 'daedalus',
    PG_USER,
    PG_PASS,
    PG_SSLMODE = 'disable'
} = process.env;

async function migrateAuth() {
    const pool = new pg.Pool({
        host: PG_HOST,
        port: parseInt(PG_PORT, 10),
        database: PG_DB,
        user: PG_USER,
        password: PG_PASS,
        ssl: PG_SSLMODE === 'require' ? { rejectUnauthorized: false } : false
    });

    const sqlPath = path.join(__dirname, 'setupAuthDB.sql');
    const sql = fs.readFileSync(sqlPath, 'utf8');

    try {
        await pool.query(sql);
        console.log('✅ Auth tables (users, auth_audit_log) created/verified.');
    } catch (err) {
        console.error('❌ Auth migration failed:', err.message);
        throw err;
    } finally {
        await pool.end();
    }
}

migrateAuth().then(() => process.exit(0)).catch(() => process.exit(1));
