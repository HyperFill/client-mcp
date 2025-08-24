// src/server.ts
import express from "express";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import HyperFillMMClient from "../client/hyper-fillmm-client";
import { config } from "../services/config";
import { registerTools } from "../core/tools/index";
import { MarketManager } from "../services/market-manager";
const app = express();
app.use(express.json());
// create server once and register tools
const server = new McpServer({ name: "example-server", version: "1.0.0" });
const hyperfillApi = new HyperFillMMClient({ account: config.account, privateKey: config.agentPrivateKey, simulationMode: true });
const marketManager = new MarketManager();
registerTools(server, marketManager);
// async function main() {
//     const transport = new StdioServerTransport();
//     await server.connect(transport);
//     // await server.connect(transport)
// }
// Start the server
async function main() {
    try {
        const transport = new StdioServerTransport();
        await server.connect(transport);
        console.error("EVM MCP Server running on stdio");
    }
    catch (error) {
        console.error("Error starting MCP server:", error);
        process.exit(1);
    }
}
main().catch((error) => {
    console.error("Fatal error in main():", error);
    process.exit(1);
});
