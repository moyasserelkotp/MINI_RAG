// Initialize MongoDB with admin user

// Get environment variables or use defaults
var rootUsername = process.env.MONGO_INITDB_ROOT_USERNAME || 'admin';
var rootPassword = process.env.MONGO_INITDB_ROOT_PASSWORD || 'admin';

// Connect to admin database
var adminDb = db.getSiblingDB('admin');

// Check if user already exists
var existingUser = adminDb.getUser(rootUsername);
if (existingUser != null) {
    print('User ' + rootUsername + ' already exists');
} else {
    // Create the root user with admin privileges
    adminDb.createUser({
        user: rootUsername,
        pwd: rootPassword,
        roles: [
            { role: 'root', db: 'admin' }
        ]
    });
    print('Created user ' + rootUsername + ' with root privileges');
}

// Verify the user was created
var user = adminDb.getUser(rootUsername);
print('User details: ' + JSON.stringify(user));

