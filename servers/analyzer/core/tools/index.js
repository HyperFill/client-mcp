import { z } from "zod";
// Schema definitions for read tool inputs
const marketSelectionSchema = z.object({
    marketName: z.string().describe("Name of the market (e.g., 'hyperfill')")
});
const getAssetSchema = z.object({
    marketName: z.string().describe("Name of the market"),
    symbol: z.string().describe("Asset symbol to get info for")
});
export function registerTools(server, marketManager) {
    // Helper function to get market client
    const getMarketClient = (marketName) => {
        const client = marketManager.getMarketClient(marketName);
        if (!client) {
            throw new Error(`Market '${marketName}' not found or not supported`);
        }
        return client;
    };
    // ===== MARKET MANAGEMENT =====
    server.registerTool("get_supported_markets", {
        title: "Get Supported Markets",
        description: "Get list of all supported markets",
    }, async () => {
        try {
            const markets = marketManager.getMarketList();
            return {
                content: [{
                        type: "text",
                        text: JSON.stringify(markets, null, 2)
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error fetching supported markets: ${err.message}`
                    }]
            };
        }
    });
    // ===== ASSET MANAGEMENT (READ) =====
    server.registerTool("fetch_market_assets", {
        title: "Fetch Market Assets",
        description: "Fetch all available trading assets for a specific market",
        inputSchema: marketSelectionSchema.shape,
    }, async ({ marketName }) => {
        try {
            const client = getMarketClient(marketName);
            const result = await client.fetchAssets();
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
                        text: `Error fetching assets for ${marketName}: ${err.message}`
                    }]
            };
        }
    });
    server.registerTool("get_market_asset", {
        title: "Get Market Asset Info",
        description: "Get detailed information about a specific asset in a market",
        inputSchema: getAssetSchema.shape,
    }, async ({ marketName, symbol }) => {
        try {
            const client = getMarketClient(marketName);
            const asset = client.getAsset(symbol);
            if (!asset) {
                return {
                    content: [{
                            type: "text",
                            text: `Asset ${symbol} not found in ${marketName}. Try fetching assets first.`
                        }]
                };
            }
            return {
                content: [{
                        type: "text",
                        text: JSON.stringify(asset, null, 2)
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error getting asset ${symbol} from ${marketName}: ${err.message}`
                    }]
            };
        }
    });
    // ===== ASSET SEARCH HELPERS =====
    server.registerTool("get_asset_by_index_token", {
        title: "Get Asset by Index Token",
        description: "Find an asset by its index token in a specific market",
        inputSchema: z.object({
            marketName: z.string().describe("Name of the market"),
            indexToken: z.string().describe("Index token to search for")
        }).shape,
    }, async ({ marketName, indexToken }) => {
        try {
            const client = getMarketClient(marketName);
            const asset = client.getAssetByIndexToken(indexToken);
            if (!asset) {
                return {
                    content: [{
                            type: "text",
                            text: `No asset found with index token ${indexToken} in ${marketName}`
                        }]
                };
            }
            return {
                content: [{
                        type: "text",
                        text: JSON.stringify(asset, null, 2)
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error finding asset by index token in ${marketName}: ${err.message}`
                    }]
            };
        }
    });
}
