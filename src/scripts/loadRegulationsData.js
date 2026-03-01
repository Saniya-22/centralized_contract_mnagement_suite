#!/usr/bin/env node

import pg from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { initializeDatabase } from './initDB.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const {
    PG_HOST = 'localhost',
    PG_PORT = '5432',
    PG_DB = 'daedalus',
    PG_USER = 'daedalus_admin',
    PG_PASSWORD = '',
    PG_PASS = '',
    REGULATIONS_NAMESPACE = 'public-regulations',
    PG_SSLMODE = 'disable'
} = process.env;

async function loadRegulationsData() {
    console.log('📚 Loading Regulations Data');
    console.log('===========================');

    console.log('Step 1: Initialize database...');
    const initResult = await initializeDatabase();

    if (initResult.denseCount > 0) {
        console.log('');
        console.log(`⚠️  Database already has ${initResult.denseCount} records.`);
        console.log('   To reload, clear the tables first:');
        console.log(`   DELETE FROM embeddings_dense WHERE namespace LIKE '${REGULATIONS_NAMESPACE}%';`);
        console.log(`   DELETE FROM embeddings_sparse WHERE namespace LIKE '${REGULATIONS_NAMESPACE}%';`);
        return initResult;
    }

    console.log('');
    console.log('Step 2: Load pre-exported data...');

    const dataFilePath = path.join(__dirname, '../../data/regulations_data.sql');

    if (!fs.existsSync(dataFilePath)) {
        console.log(`⚠️  Data file not found at ${dataFilePath}`);
        console.log('   Run ingestion instead: node src/scripts/ingestRegulations.js');
        return initResult;
    }

    const pool = new pg.Pool({
        host: PG_HOST,
        port: parseInt(PG_PORT),
        database: PG_DB,
        user: PG_USER,
        password: PG_PASSWORD || PG_PASS,
        ssl: PG_SSLMODE === 'require' ? { rejectUnauthorized: false } : false
    });

    try {
        const client = await pool.connect();

        console.log('Reading data file...');
        const dataSql = fs.readFileSync(dataFilePath, 'utf8');
        const fileSize = (fs.statSync(dataFilePath).size / 1024 / 1024).toFixed(2);
        console.log(`File size: ${fileSize} MB`);

        console.log('Executing SQL (this may take a moment)...');
        const startTime = Date.now();

        await client.query(dataSql);

        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log(`✅ Data loaded in ${duration}s`);

        console.log('');
        console.log('Step 3: Verify data...');
        const denseCount = await client.query('SELECT COUNT(*) FROM embeddings_dense');
        const sparseCount = await client.query('SELECT COUNT(*) FROM embeddings_sparse');
        console.log(`Dense embeddings: ${denseCount.rows[0].count}`);
        console.log(`Sparse embeddings: ${sparseCount.rows[0].count}`);

        const namespaces = await client.query(`
            SELECT namespace, COUNT(*) as count
            FROM embeddings_dense
            GROUP BY namespace
        `);
        console.log('');
        console.log('Namespaces:');
        namespaces.rows.forEach(row => {
            console.log(`  - ${row.namespace}: ${row.count} chunks`);
        });

        client.release();

        console.log('');
        console.log('===========================');
        console.log('✅ Data load complete!');

        return {
            success: true,
            denseCount: parseInt(denseCount.rows[0].count),
            sparseCount: parseInt(sparseCount.rows[0].count)
        };
    } catch (error) {
        console.error('❌ Data load failed:', error.message);
        throw error;
    } finally {
        await pool.end();
    }
}

loadRegulationsData()
    .then(result => {
        console.log('');
        console.log('Test with:');
        console.log('  curl "http://localhost:8000/cai/api/chat/bot_response?query=What%20is%20FAR%2052.236-2"');
        process.exit(0);
    })
    .catch(error => {
        console.error('❌ loadRegulationsData FATAL ERROR:', error.message);
        console.error(error.stack);
        process.exit(1);
    });

export { loadRegulationsData };
