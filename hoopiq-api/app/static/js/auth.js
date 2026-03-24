/* Basketball Shot Tracker App
   Developed by Christopher Hong
   Team Name: HoopIQ
   Team Members: Christopher Hong, Alfonso Mejia Vasquez, Gondra Kelly, Matthew Margulies, Carlos Orozco
   Start Web Development Date: October 2025
   Finished Web Development Date: June 2026 (Ideally)
   static/js/auth.js - Handles user authentication and session management
*/

// =============================
// Local Storage Helpers
// =============================


function getUsers() {
    const users = localStorage.getItem("users");
    return users ? JSON.parse(users) : {};
}

function getSession() {
    return localStorage.getItem("session_user");
}

function saveSession(username) {
    localStorage.setItem("session_user", username);
}

function clearSession() {
    localStorage.removeItem("session_user");
}

// =============================
// Modal Controls
// =============================

function openModal() {
    const modal = document.getElementById("auth-modal");
    if (modal) modal.style.display = "flex";
}

function closeModal() {
    const modal = document.getElementById("auth-modal");
    if (modal) modal.style.display = "none";
}

function switchTab(tab) {
    const loginTab = document.getElementById("login-tab");
    const registerTab = document.getElementById("register-tab");

    const loginForm = document.getElementById("login-form");
    const registerForm = document.getElementById("register-form");

    if (!loginTab || !registerTab || !loginForm || !registerForm) return;

    if (tab === "login") {
        loginTab.classList.add("active");
        registerTab.classList.remove("active");

        loginForm.style.display = "block";
        registerForm.style.display = "none";
    } else {
        registerTab.classList.add("active");
        loginTab.classList.remove("active");

        loginForm.style.display = "none";
        registerForm.style.display = "block";
    }
}


// =============================
// Login
// =============================

function handleLogin() {
    const usernameInput = document.getElementById("login-username");
    const passwordInput = document.getElementById("login-password");

    if (!usernameInput || !passwordInput) return;

    const username = usernameInput.value.trim();
    const password = passwordInput.value.trim();

    const users = getUsers();

    if (!users[username] || users[username] !== password) {
        alert("Invalid username or password.");
        return;
    }

    saveSession(username);
    updateAuthUI();
    closeModal();
}


// =============================
// Register
// =============================

function handleRegister() {
    const usernameInput = document.getElementById("register-username");
    const passwordInput = document.getElementById("register-password");

    if (!usernameInput || !passwordInput) return;

    const username = usernameInput.value.trim();
    const password = passwordInput.value.trim();

    if (!username || !password) {
        alert("Please fill out all fields.");
        return;
    }

    const users = getUsers();

    if (users[username]) {
        alert("User already exists.");
        return;
    }

    users[username] = password;
    saveUsers(users);

    alert("Account created. You can now log in.");

    switchTab("login");
}


// =============================
// Logout
// =============================

function handleLogout() {
    clearSession();
    updateAuthUI();
}


// =============================
// Update Auth UI
// =============================

function updateAuthUI() {
    const user = getSession();

    const loginBtn = document.getElementById("login-btn");
    const logoutBtn = document.getElementById("logout-btn");
    const userLabel = document.getElementById("user-label");

    if (!loginBtn || !logoutBtn || !userLabel) return;

    if (user) {
        loginBtn.style.display = "none";
        logoutBtn.style.display = "inline-block";
        userLabel.textContent = user;
    } else {
        loginBtn.style.display = "inline-block";
        logoutBtn.style.display = "none";
        userLabel.textContent = "Guest";
    }
}