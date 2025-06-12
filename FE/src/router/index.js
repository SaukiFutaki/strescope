import { createRouter, createWebHistory } from "vue-router";
import BerandaView from "../views/BerandaView.vue";
import CekStresView from "../views/CekStresView.vue";
import RiwayatView from "../views/RiwayatView.vue";
import HitungBPMView from "../views/HitungBPMView.vue";
import LoginView from "../views/LoginView.vue";
import RegisterView from "../views/RegisterView.vue";
import HasilView from "../views/HasilView.vue";
import { useAuthStore } from "../stores/authStore";

const router = createRouter({
  history: createWebHistory("/"),
  routes: [
    {
      path: "/",
      name: "home",
      component: BerandaView,
      meta: {
        layout: "main",
        requiresAuth: true,
      },
    },
    {
      path: "/cek-stres",
      name: "cek-stres",
      component: CekStresView,
      meta: {
        layout: "main",
        requiresAuth: true,
      },
    },
    {
      path: "/hasil",
      name: "hasil",
      component: HasilView,
      meta: {
        layout: "main",
        requiresAuth: true,
      },
    },
    {
      path: "/riwayat",
      name: "riwayat",
      component: RiwayatView,
      meta: {
        layout: "main",
        requiresAuth: true,
      },
    },
    {
      path: "/hitung-bpm",
      name: "hitung-bpm",
      component: HitungBPMView,
      meta: {
        layout: "main",
        requiresAuth: true,
      },
    },
    {
      path: "/login",
      name: "login",
      component: LoginView,
      meta: {
        layout: "auth",
        requiresAuth: false,
      },
    },
    {
      path: "/register",
      name: "register",
      component: RegisterView,
      meta: {
        layout: "auth",
        requiresAuth: false,
      },
    },
  ],
});

// router.beforeEach(async (to, from, next) => {
//   const authStore = useAuthStore();

//   const isAuthenticated = authStore.isLoggedIn;
//   const requiresAuth = to.matched.some((record) => record.meta.requiresAuth);

//   if (requiresAuth && !isAuthenticated) {
//     next({ name: "login" });
//   } else {
//     next();
//   }
// });

router.afterEach(async (to, from, failure) => {
  setTimeout(() => {
    if (
      window.HSStaticMethods &&
      typeof window.HSStaticMethods.autoInit === "function"
    ) {
      window.HSStaticMethods.autoInit();
    }
  }, 100);
});

export default router;
