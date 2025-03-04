const api = axios.create({
  baseURL: "https://teamsynthbe.onrender.com/api",
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  withCredentials: true
}); 