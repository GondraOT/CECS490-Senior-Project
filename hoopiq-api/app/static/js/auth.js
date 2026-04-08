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

function saveUsers(users) {
    localStorage.setItem("users", JSON.stringify(users));
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

function getUserData() {
    const data = localStorage.getItem("user_data");
    return data ? JSON.parse(data) : {};
}

function saveUserData(data) {
    localStorage.setItem("user_data", JSON.stringify(data));
}

function getCurrentUserData() {
    const user = getSession();
    if (!user) return null;

    const data = getUserData();
    return data[user] || null;
}

function setCurrentUserData(shots, stats) {
    const user = getSession();
    if (!user) return;

    const data = getUserData();

    data[user] = {
        shots: shots || [],
        stats: {
        attempts: stats?.attempts ?? 0,
        makes: stats?.makes ?? 0
        }
    };

    saveUserData(data);
}

// =============================
// Modal Controls
// =============================
function openModal(tab = "login") {
    const modal = document.getElementById("auth-modal");
    if (modal) {
        modal.classList.add("active");
        switchTab(tab);
    }
}

function closeModal() {
    const modal = document.getElementById("auth-modal");
    if (modal) modal.classList.remove("active");
}

function switchTab(tab) {
    const loginTab = document.getElementById("tab-login");
    const registerTab = document.getElementById("tab-register");

    const loginForm = document.getElementById("form-login");
    const registerForm = document.getElementById("form-register");

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
    const emailInput = document.getElementById("login-email");
    const passwordInput = document.getElementById("login-password");

    if (!emailInput || !passwordInput) {
        alert("Please enter email and password.");
        return;
    }

    const email = emailInput.value.trim();
    const password = passwordInput.value.trim();

    const users = getUsers();

    if (!users[email] || users[email] !== password) {
        alert("Invalid email or password.");
        return;
    }

    saveSession(email);
    updateAuthUI();
    closeModal();

    const allUserData = getUserData();

    if (!allUserData[email]) {
        allUserData[email] = {
            shots: [],
            stats: { attempts: 0, makes: 0 }
        };
        saveUserData(allUserData);
    }

    loadUserSessionData(email);
}


// =============================
// Register
// =============================
function handleRegister() {
    const nameInput = document.getElementById("reg-name");
    const emailInput = document.getElementById("reg-email");
    const passwordInput = document.getElementById("reg-password");
    const confirmInput = document.getElementById("reg-confirm");

    if (!nameInput || !emailInput || !passwordInput || !confirmInput) return;

    const name = nameInput.value.trim();
    const email = emailInput.value.trim();
    const password = passwordInput.value.trim();
    const confirm = confirmInput.value.trim();

    if (!name || !email || !password || !confirm) {
        alert("Please fill out all fields.");
        return;
    }

    if (password !== confirm) {
        alert("Passwords do not match.");
        return;
    }

    const users = getUsers();

    if (users[email]) {
        alert("User already exists.");
        return;
    }

    users[email] = password;

    // Also initialize user data
    const allUserData = getUserData();

    if (!allUserData[email]) {
        allUserData[email] = {
            shots: [],
            stats: { attempts: 0, makes: 0 }
        };
    }

    saveUsers(users);
    saveUserData(allUserData);

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

    const authButtons = document.getElementById("auth-buttons");
    const userInfo = document.getElementById("user-info");
    const userGreeting = document.getElementById("user-greeting");

    if (!authButtons || !userInfo || !userGreeting) return;

    if (user) {
        // Show logged-in UI
        authButtons.style.display = "none";
        userInfo.style.display = "flex";
        userGreeting.textContent = user;
    } else {
        // Show guest UI
        authButtons.style.display = "flex";
        userInfo.style.display = "none";
        userGreeting.textContent = "";
    }
}

// =============================
// Clear All Auth Data (For Testing)
// =============================
function clearAllAuthData() {
    localStorage.removeItem("users");
    localStorage.removeItem("session_user");
    console.log("Auth data cleared.");
}

document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("auth-modal");
    const closeBtn = document.getElementById("modal-close");

    // Close button (✕)
    if (closeBtn) {
        closeBtn.addEventListener("click", closeModal);
    }

    // Click outside modal closes it
    if (modal) {
        modal.addEventListener("mousedown", (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
    }

    updateAuthUI();
});

document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;

    const loginForm = document.getElementById("form-login");
    const registerForm = document.getElementById("form-register");

    const loginVisible = window.getComputedStyle(loginForm).display !== "none";
    const registerVisible = window.getComputedStyle(registerForm).display !== "none";

    if (loginVisible) {
        handleLogin();
    } else if (registerVisible) {
        handleRegister();
    }
});

window.clearAllAuthData = clearAllAuthData;
window.getUserData = getUserData;
window.saveUserData = saveUserData;
window.getCurrentUserData = getCurrentUserData;
window.setCurrentUserData = setCurrentUserData;
window.getUsers = getUsers;