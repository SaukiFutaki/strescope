const Hapi = require("@hapi/hapi");
const { createClient } = require("@libsql/client");
const Joi = require("@hapi/joi");
const JWT = require("@hapi/jwt");
const bcrypt = require("bcrypt");
require("dotenv").config();
const axios = require("axios");

const physicalSchema = Joi.object({
  gender: Joi.string().valid("male", "female").required(),
  age: Joi.number().integer().min(0).required(),
  occupation: Joi.string().required(),
  sleep_duration: Joi.number().min(0).max(24).required(),
  sleep_quality: Joi.number().integer().min(1).max(10).required(),
  bmi_category: Joi.string()
    .valid("underweight", "normal", "overweight", "obese")
    .required(),
  blood_pressure: Joi.string()
    .pattern(/^\d+\/\d+$/)
    .required(),
  heart_rate: Joi.number().integer().min(0).required(),
  daily_steps: Joi.number().integer().min(0).required(),
  sleep_disorder: Joi.string().allow("none", "Insomnia").required(),
});

const stressResponseSchema = Joi.object({
  responses: Joi.object()
    .pattern(Joi.string(), Joi.number().integer().min(0).max(4))
    .required(),
}).unknown();

const init = async () => {
  const db = createClient({
    url: process.env.TURSO_DATABASE_URL,
    authToken: process.env.TURSO_AUTH_TOKEN,
  });

  const server = Hapi.server({
    port: 3000,
    host: "localhost",
  });

  await server.register({
    plugin: require("hapi-cors"),
    options: {
      origins: ["http://localhost:8081", "http://localhost:8081"],
      headers: ["Accept", "Authorization", "Content-Type", "If-None-Match"],
      methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    },
  });

  await server.register(JWT);

  server.auth.strategy("jwt", "jwt", {
    keys: process.env.JWT_SECRET,
    verify: {
      aud: "urn:audience:hapi-jwt",
      iss: "urn:issuer:hapi-jwt",
      sub: false,
    },
    validate: (artifacts, request, h) => {
      console.log("JWT Validation:", {
        decoded: artifacts.decoded,
        credentials: artifacts.decoded.payload,
      });
      return {
        isValid: true,
        credentials: {
          id: artifacts.decoded.payload.id,
          role: artifacts.decoded.payload.role,
        },
      };
    },
  });

  const userSchema = Joi.object({
    username: Joi.string().min(3).required(),
    email: Joi.string().email().required(),
    password: Joi.string().min(6).required(),
    role: Joi.string().valid("admin", "user").default("user"),
  });

  const validateAuth = (request, h) => {
    const auth = request.auth;
    console.log("Validate Auth:", { isAuthenticated: auth.isAuthenticated });
    if (!auth.isAuthenticated) {
      return h.response({ error: "Authentication required" }).code(401);
    }
    return h.continue;
  };

  server.route({
    method: "POST",
    path: "/register",
    options: {
      validate: {
        payload: userSchema,
        failAction: (request, h, err) => {
          return h.response({ error: err.message }).code(400).takeover();
        },
      },
    },
    handler: async (request, h) => {
      const { username, email, password, role } = request.payload;
      const hashedPassword = await bcrypt.hash(password, 10);

      await db.execute({
        sql: "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
        args: [username, email, hashedPassword, role],
      });

      return h.response({ message: "User registered successfully" }).code(201);
    },
  });

  server.route({
    method: "POST",
    path: "/login",
    options: {
      validate: {
        payload: Joi.object({
          email: Joi.string().email().required(),
          password: Joi.string().min(6).required(),
        }),
      },
    },
    handler: async (request, h) => {
      const { email, password } = request.payload;
      const user = await db.execute({
        sql: "SELECT * FROM users WHERE email = ?",
        args: [email],
      });

      if (user.rows.length === 0) {
        return h.response({ message: "Invalid email or password" }).code(401);
      }

      const isValid = await bcrypt.compare(password, user.rows[0].password);
      if (!isValid) {
        return h.response({ message: "Invalid email or password" }).code(401);
      }

      const token = JWT.token.generate(
        {
          id: user.rows[0].id,
          email: user.rows[0].email,
          role: user.rows[0].role,
        },
        {
          key: process.env.JWT_SECRET,
          algorithm: "HS256",
        },
        {
          aud: "urn:audience:hapi-jwt",
          iss: "urn:issuer:hapi-jwt",
        }
      );

      console.log("Generated Token:", token);

      return h
        .response({
          user: {
            id: user.rows[0].id,
            email: user.rows[0].email,
            username: user.rows[0].username,
            role: user.rows[0].role,
          },
          token,
        })
        .code(200);
    },
  });

  server.route({
    method: "GET",
    path: "/users",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
      validate: {
        headers: Joi.object({
          authorization: Joi.string().required(),
        }).unknown(),
      },
    },
    handler: async (request, h) => {
      const auth = request.auth.credentials;
      if (!auth || auth.role !== "admin") {
        return h.response({ error: "Admin access required" }).code(403);
      }

      const result = await db.execute(
        "SELECT id, username, email, role FROM users"
      );
      return h.response(result.rows).code(200);
    },
  });

  server.route({
    method: "GET",
    path: "/users/{id}",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
      validate: {
        params: Joi.object({
          id: Joi.number().integer().required(),
        }),
        headers: Joi.object({
          authorization: Joi.string().required(),
        }).unknown(),
      },
    },
    handler: async (request, h) => {
      const { id } = request.params;
      const result = await db.execute({
        sql: "SELECT id, username, email, role FROM users WHERE id = ?",
        args: [id],
      });
      const user = result.rows[0];

      if (!user) {
        return h.response({ error: "User not found" }).code(404);
      }

      return h.response(user).code(200);
    },
  });

  server.route({
    method: "PUT",
    path: "/users/{id}",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
      validate: {
        params: Joi.object({
          id: Joi.number().integer().required(),
        }),
        payload: Joi.object({
          username: Joi.string().min(3),
          email: Joi.string().email(),
          role: Joi.string().valid("admin", "user"),
        }).min(1),
        headers: Joi.object({
          authorization: Joi.string().required(),
        }).unknown(),
      },
    },
    handler: async (request, h) => {
      const { id } = request.params;
      const { username, email, role } = request.payload;
      const auth = request.auth.credentials;

      if (!auth || (auth.role !== "admin" && id != auth.id)) {
        return h
          .response({ error: "Unauthorized to update this user" })
          .code(403);
      }

      const updates = [];
      const args = [];
      if (username) {
        updates.push("username = ?");
        args.push(username);
      }
      if (email) {
        updates.push("email = ?");
        args.push(email);
      }
      if (role) {
        updates.push("role = ?");
        args.push(role);
      }
      args.push(id);

      if (updates.length === 0) {
        return h.response({ error: "No updates provided" }).code(400);
      }

      await db.execute({
        sql: `UPDATE users SET ${updates.join(", ")} WHERE id = ?`,
        args: args,
      });

      return h.response({ message: "User updated successfully" }).code(200);
    },
  });

  server.route({
    method: "DELETE",
    path: "/users/{id}",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
      validate: {
        params: Joi.object({
          id: Joi.number().integer().required(),
        }),
        headers: Joi.object({
          authorization: Joi.string().required(),
        }).unknown(),
      },
    },
    handler: async (request, h) => {
      const { id } = request.params;
      const auth = request.auth.credentials;

      if (!auth || (auth.role !== "admin" && id != auth.id)) {
        return h
          .response({ error: "Unauthorized to delete this user" })
          .code(403);
      }

      await db.execute({
        sql: "DELETE FROM users WHERE id = ?",
        args: [id],
      });

      return h.response({ message: "User deleted successfully" }).code(200);
    },
  });

  server.route({
    method: "POST",
    path: "/predict/physical",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
      validate: {
        payload: physicalSchema,
        headers: Joi.object({
          authorization: Joi.string().required(),
        }).unknown(),
      },
    },
    handler: async (request, h) => {
      const {
        gender,
        age,
        occupation,
        sleep_duration,
        sleep_quality,
        bmi_category,
        blood_pressure,
        heart_rate,
        daily_steps,
        sleep_disorder,
      } = request.payload;
      const user_id = request.auth.credentials.id;

      const result = await db.execute({
        sql: "INSERT INTO physical_stress (user_id, gender, age, occupation, sleep_duration, sleep_quality, bmi_category, blood_pressure, heart_rate, daily_steps, sleep_disorder, stress_level) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        args: [
          user_id,
          gender,
          age,
          occupation,
          sleep_duration,
          sleep_quality,
          bmi_category,
          blood_pressure,
          heart_rate,
          daily_steps,
          sleep_disorder,
          "pending",
        ],
      });

      return h
        .response({
          message: "Physical stress data saved",
          id: result.lastInsertRowid,
        })
        .code(201);
    },
  });

  server.route({
    method: "GET",
    path: "/predict/physical",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
    },
    handler: async (request, h) => {
      const user_id = request.auth.credentials.id;
      const result = await db.execute({
        sql: "SELECT * FROM physical_stress WHERE user_id = ?",
        args: [user_id],
      });
      return h.response(result.rows).code(200);
    },
  });

  server.route({
    method: "GET",
    path: "/questions",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
    },
    handler: async (request, h) => {
      try {
        console.log("Requesting questions from ML server...");
        const response = await axios.get("http://127.0.0.1:5000/questions");
        console.log("Flask response:", response.data);
        return h.response(response.data).code(200);
      } catch (error) {
        console.error("Error fetching questions:", error.message);
        return h
          .response({
            error: "Failed to fetch questions from prediction service",
          })
          .code(500);
      }
    },
  });

  server.route({
    method: "POST",
    path: "/predict/stress",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
      validate: {
        payload: stressResponseSchema,
        headers: Joi.object({
          authorization: Joi.string().required(),
        }).unknown(),
      },
    },
    handler: async (request, h) => {
      try {
        const { responses } = request.payload;
        const user_id = request.auth.credentials.id;

        const flaskResponse = await axios.post(
          "http://127.0.0.1:5000/predict",
          { responses }
        );

        const result = await db.execute({
          sql: `
            INSERT INTO stress_predictions (
              user_id, stress_level, stress_level_numeric, confidence,
              low_probability, moderate_probability, high_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
          `,
          args: [
            user_id,
            flaskResponse.data.prediction.stress_level,
            flaskResponse.data.prediction.stress_level_numeric,
            flaskResponse.data.prediction.confidence,
            flaskResponse.data.prediction.probabilities.Low,
            flaskResponse.data.prediction.probabilities.Moderate,
            flaskResponse.data.prediction.probabilities.High,
          ],
        });

        return h
          .response({
            id: result.lastInsertRowid,
            prediction: flaskResponse.data.prediction,
          })
          .code(200);
      } catch (error) {
        console.error("Prediction error:", error.message);
        return h
          .response({ error: "Prediction failed: " + error.message })
          .code(500);
      }
    },
  });

  server.route({
    method: "GET",
    path: "/results/{physical_id}/{stress_id}",
    options: {
      auth: "jwt",
      pre: [{ method: validateAuth }],
      validate: {
        params: Joi.object({
          physical_id: Joi.number().integer().required(),
          stress_id: Joi.number().integer().required(),
        }),
        headers: Joi.object({
          authorization: Joi.string().required(),
        }).unknown(),
      },
    },
    handler: async (request, h) => {
      const { physical_id, stress_id } = request.params;
      const user_id = request.auth.credentials.id;

      const physicalResult = await db.execute({
        sql: "SELECT * FROM physical_stress WHERE id = ? AND user_id = ?",
        args: [physical_id, user_id],
      });

      const stressResult = await db.execute({
        sql: "SELECT * FROM stress_predictions WHERE id = ? AND user_id = ?",
        args: [stress_id, user_id],
      });

      if (physicalResult.rows.length === 0 || stressResult.rows.length === 0) {
        return h.response({ error: "Result not found" }).code(404);
      }

      return h
        .response({
          physical: physicalResult.rows[0],
          stress: stressResult.rows[0],
        })
        .code(200);
    },
  });

  await server.start();
  console.log("Server running at:", server.info.uri);
};

init().catch((err) => {
  console.error(err);
  process.exit(1);
});
