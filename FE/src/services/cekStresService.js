import { getData, postData } from "@/api";

const postCekStres = async (data) => {
  const token = localStorage.getItem("token");

  const response = await postData("/cek-stres", {
    data,
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  return response.data;
};

const getRiwayatCek = async () => {
  const token = localStorage.getItem("token");

  const response = await getData("/riwayat", {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });
  return response.data;
};

export { postCekStres, getRiwayatCek };
