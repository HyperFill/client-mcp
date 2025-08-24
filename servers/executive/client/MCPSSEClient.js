// import { Client } from "@modelcontextprotocol/sdk/client/index";
// import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
// import { Client } from "@modelcontextprotocol/sdk";
// import { StdioClientTransport } from "@modelcontextprotocol/sdk";
// import { Client } from "@modelcontextprotocol/sdk/dist/index.js";
// import { StdioClientTransport } from "@modelcontextprotocol/sdk/dist/index.js";
// import { Client } from "@modelcontextprotocol/sdk/dist/esm/client/index.js";
// import { StdioClientTransport } from "@modelcontextprotocol/sdk/dist/esm/client/stdio.js";
// import { Client, StdioClientTransport } from "@modelcontextprotocol/sdk";
export async function fetchMcpSeiClient() {
    const transport = new StdioClientTransport({
        command: "node",
        args: ["C:/Users/norbe/Documents/Hackathons/Sei Hackathon/HyperFill/sei-mcp-server/build/index.js"]
    });
    const client = new Client({
        name: "example-client",
        version: "1.0.0"
    });
    await client.connect(transport);
    return client;
}
