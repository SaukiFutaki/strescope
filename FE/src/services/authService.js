import { postData } from "@/api";
import { useAuthStore } from "../stores/authStore";

const login = async (credentials) => {
  const response = await postData("login", { data: credentials });

  const authStore = useAuthStore();

  const { user, token } = response.data;

  const loginSuccess = authStore.login(user, token);

  return {
    success: loginSuccess,
    user: authStore.getUser,
    token: authStore.getToken,
  };
};

const logout = () => {
  const authStore = useAuthStore();
  const token = authStore.getToken;

  if (token) {
    getData("logout", {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
  }

  const logoutSuccess = authStore.logout();

  return {
    success: logoutSuccess,
    message: "Logout successful",
  };
};

export { login, logout };
