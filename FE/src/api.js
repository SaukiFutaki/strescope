const BASE_URL = "https://somewhere.com";

export async function getData(endpoint, { headers = {} } = {}) {
  const fetchResponse = await fetch(`${BASE_URL}/${endpoint}`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      ...headers,
    },
  });
  return await fetchResponse.json();
}

export async function postData(endpoint, { data, headers = {} } = {}) {
  // Prepare default headers
  const defaultHeaders = {};
  
  // Only set Content-Type if not already provided in custom headers
  // and if data is not FormData
  if (!headers["Content-Type"] && !headers["content-type"] && !(data instanceof FormData)) {
    defaultHeaders["Content-Type"] = "application/json";
  }
  
  // Prepare body based on data type
  let body;
  if (data instanceof FormData) {
    body = data; // FormData is sent as-is
  } else if (typeof data === 'object' && data !== null) {
    body = JSON.stringify(data); // Objects are stringified
  } else {
    body = data; // Strings, etc. are sent as-is
  }

  const fetchResponse = await fetch(`${BASE_URL}/${endpoint}`, {
    method: "POST",
    headers: {
      ...defaultHeaders,
      ...headers, // Custom headers override defaults
    },
    body,
  });
  
  return await fetchResponse.json();
}

export async function putData(endpoint, { data, headers = {} } = {}) {
  const defaultHeaders = {};
  
  if (!headers["Content-Type"] && !headers["content-type"] && !(data instanceof FormData)) {
    defaultHeaders["Content-Type"] = "application/json";
  }
  
  let body;
  if (data instanceof FormData) {
    body = data;
  } else if (typeof data === 'object' && data !== null) {
    body = JSON.stringify(data);
  } else {
    body = data;
  }

  const fetchResponse = await fetch(`${BASE_URL}/${endpoint}`, {
    method: "PUT",
    headers: {
      ...defaultHeaders,
      ...headers,
    },
    body,
  });
  
  return await fetchResponse.json();
}

export async function deleteData(endpoint, { data, headers = {} } = {}) {
  const defaultHeaders = {};
  
  if (data && !headers["Content-Type"] && !headers["content-type"] && !(data instanceof FormData)) {
    defaultHeaders["Content-Type"] = "application/json";
  }
  
  let body;
  if (data) {
    if (data instanceof FormData) {
      body = data;
    } else if (typeof data === 'object' && data !== null) {
      body = JSON.stringify(data);
    } else {
      body = data;
    }
  }

  const fetchResponse = await fetch(`${BASE_URL}/${endpoint}`, {
    method: "DELETE",
    headers: {
      ...defaultHeaders,
      ...headers,
    },
    body,
  });
  
  return await fetchResponse.json();
}