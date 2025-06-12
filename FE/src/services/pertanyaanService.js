import { getData } from "@/api";

const getPertanyaan = async () => {
  const token = localStorage.getItem("token");

  const response = await getData("/pertanyaan", {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  return response.data;
};

export { getPertanyaan };
