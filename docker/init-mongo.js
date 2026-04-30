// Note: MongoDB automatically creates the root user via MONGO_INITDB_ROOT_USERNAME
// and MONGO_INITDB_ROOT_PASSWORD environment variables during container startup.
// This file is kept for reference but not needed for basic setup.
//
// If you need to create additional users or collections, uncomment and modify below:

// Example: Create an application database and user
/*
db = db.getSiblingDB('mini_rag_db');
db.createCollection('projects');
db.createCollection('chat_sessions');
db.createCollection('data_chunks');
db.createCollection('chat_messages');
db.createCollection('assets');

// Create app-specific user (optional)
db.createUser({
  user: 'app_user',
  pwd: 'app_password',
  roles: [{ role: 'readWrite', db: 'mini_rag_db' }]
});

print('MongoDB initialization complete');
*/

