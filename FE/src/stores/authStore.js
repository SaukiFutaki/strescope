import { defineStore } from "pinia";
import User from "@/models/User.js";

export const useAuthStore = defineStore("auth", {
  state: () => ({
    user: null,
    token: null,
    isAuthenticated: false,
  }),

  getters: {
    getUser: (state) => state.user,

    getToken: (state) => state.token,

    isLoggedIn: (state) => state.isAuthenticated && state.token !== null,

    getUserName: (state) => state.user?.name || "",

    getUserEmail: (state) => state.user?.email || "",

    getUserId: (state) => state.user?.id || null,
  },

  actions: {
    login(userData, token) {
      try {
        this.user = new User({
          id: userData.id,
          name: userData.name,
          email: userData.email,
        });

        this.token = token;
        this.isAuthenticated = true;

        localStorage.setItem("token", this.token || "");

        return true;
      } catch (error) {
        console.error("Error during login:", error);
        return false;
      }
    },

    logout() {
      this.user = null;
      this.token = null;
      this.isAuthenticated = false;

      localStorage.removeItem("token");
    },

    checkAuthStatus() {
      if (this.token && this.user) {
        return this.isAuthenticated;
      }
      return false;
    },
  },
});
