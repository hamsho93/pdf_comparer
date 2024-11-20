import axios from "axios";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export const uploadDocuments = async (formData) => {
  const response = await axios.post(`${API_BASE_URL}/upload`, formData);
  return response.data;
};

export const getDiffResults = async () => {
  const response = await axios.get(`${API_BASE_URL}/diff`);
  return response.data;
};