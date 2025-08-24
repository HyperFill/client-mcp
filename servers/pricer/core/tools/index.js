import { z } from "zod";
import { getMidPriceGate } from "../../client/price-oracle-client";
// Schema definitions for read tool inputs
const fetchOraclePriceSchema = z.object({
    base: z.string().describe("This is the base currency of the asset pair"),
    quote: z.string().describe("this is the quote currency of the asset pair")
});
const hyperfillAbi = [
    {
        "inputs": [],
        "name": "totalSupply",
        "outputs": [
            {
                "internalType": "uint256",
                "name": "",
                "type": "uint256"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
];
export function registerTools(server, filamentApi, seiClientFactory) {
    // ===== ASSET MANAGEMENT (READ) =====
    server.registerTool("fetch_vault_asset_balance", {
        title: "vault balance",
        description: "Fetches Vault Available Balance",
    }, async () => {
        try {
            const client = await seiClientFactory();
            const result = await client.callTool({
                name: "read_contract",
                arguments: {
                    contractAddress: "0xbaC8D6A511A673fCE111D8c14c760aDE68116558",
                    abi: hyperfillAbi,
                    functionName: "totalSupply",
                    args: [],
                    network: "sei-testnet",
                }
            });
            return {
                content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error fetching assets: ${err.message}`
                    }]
            };
        }
    });
    server.registerTool("fetch_oracle_price", {
        title: "Fetch Oracle Price",
        description: "Fetch Oracle Price",
        inputSchema: fetchOraclePriceSchema.shape
    }, async ({ base, quote }) => {
        try {
            const result = await getMidPriceGate({ base, quote });
            return {
                content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error fetching assets: ${err.message}`
                    }]
            };
        }
    });
    // server.registerTool(
    //     "fetch_vault_asset_balance",
    //     {
    //         title: "Fetch Vault Asset Balance",
    //         description: "Fetches Vault Available Balance",
    //     },
    //     async () => {
    //         try {
    //             const client = await seiClientFactory()
    //             const result = await client.callTool({
    //                 name: "read_contract",
    //                 arguments: {
    //                     contractAddress: "0xbaC8D6A511A673fCE111D8c14c760aDE68116558",
    //                     abi: hyperfillAbi,
    //                     functionName: "totalSupply",
    //                     args: [],
    //                     network: "sei-testnet",
    //                 }
    //             });
    //             return {
    //                 content: [{
    //                     type: "text",
    //                     text: JSON.stringify(result, null, 2)
    //                 }]
    //             };
    //         } catch (err: any) {
    //             return {
    //                 content: [{
    //                     type: "text",
    //                     text: `Error fetching assets: ${err.message}`
    //                 }]
    //             };
    //         }
    //     }
    // );
}
