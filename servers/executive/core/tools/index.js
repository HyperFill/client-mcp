import { z } from "zod";
import { MarketMakerBotClient } from "../../client/hyperfill-mm-bot-client.js";
import { config } from "../../services/config.js";
import { getMidPriceGate } from "../../client/price-oracle-client.js";
// Enhanced schema definitions
const moveAssetAmountSchema = z.object({
    assetAmountToMove: z.string().describe("Asset amount to move from vault"),
});
const botConfigSchema = z.object({
    baseAsset: z.string().describe("Base asset symbol (e.g., SEI)"),
    quoteAsset: z.string().describe("Quote asset symbol (e.g., USDT)"),
    quantity: z.number().describe("Order quantity"),
    side: z.enum(["bid", "ask"]).describe("Order side - bid or ask"),
    spreadPercentage: z.number().optional().describe("Spread percentage (default 0.5%)"),
    referencePrice: z.number().optional().describe("Manual reference price override"),
});
const botModifySchema = z.object({
    spreadPercentage: z.number().optional().describe("New spread percentage"),
    quantity: z.number().optional().describe("New order quantity"),
    referencePrice: z.number().optional().describe("New reference price"),
});
const hyperfillOrderBookAbi = [
    {
        "inputs": [
            { "internalType": "uint256", "name": "amount", "type": "uint256" },
            { "internalType": "address", "name": "tradingWallet", "type": "address" }
        ],
        "name": "moveFromVaultToWallet",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            { "internalType": "uint256", "name": "amount", "type": "uint256" },
            { "internalType": "uint256", "name": "profitAmount", "type": "uint256" },
            { "internalType": "address", "name": "fromWallet", "type": "address" }
        ],
        "name": "moveFromWalletToVault",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
];
export function registerTools(server, filamentApi, seiClientFactory) {
    // Initialize bot client
    const botClient = new MarketMakerBotClient();
    // ===== ASSET MANAGEMENT TOOLS =====
    server.registerTool("move_assets_from_vault_to_wallet", {
        title: "Move to Agent Wallet",
        description: "Move a particular asset amount from vault to agent wallet",
        inputSchema: moveAssetAmountSchema.shape,
    }, async ({ assetAmountToMove }) => {
        try {
            const client = await seiClientFactory();
            const result = await client.callTool({
                name: "write_contract",
                arguments: {
                    contractAddress: config.vaultContractAddress,
                    abi: hyperfillOrderBookAbi,
                    functionName: "moveFromVaultToWallet",
                    args: [assetAmountToMove, config.agentWallet],
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
                        text: `Error moving assets from vault: ${err.message}`
                    }]
            };
        }
    });
    server.registerTool("move_assets_from_wallet_to_vault", {
        title: "Move to Vault",
        description: "Move a particular asset amount from agent wallet to vault",
        inputSchema: moveAssetAmountSchema.shape,
    }, async ({ assetAmountToMove }) => {
        try {
            const client = await seiClientFactory();
            const result = await client.callTool({
                name: "write_contract",
                arguments: {
                    contractAddress: config.vaultContractAddress,
                    abi: hyperfillOrderBookAbi,
                    functionName: "moveFromWalletToVault",
                    args: [assetAmountToMove, "0", config.agentWallet], // Added missing profitAmount parameter
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
                        text: `Error moving assets to vault: ${err.message}`
                    }]
            };
        }
    });
    // ===== MARKET MAKER BOT CONTROL TOOLS =====
    server.registerTool("start_market_maker_bot", {
        title: "Start Market Maker Bot",
        description: "Start the market maker bot with specified configuration",
        inputSchema: botConfigSchema.shape,
    }, async ({ baseAsset, quoteAsset, quantity, side, spreadPercentage = 0.5, referencePrice }) => {
        try {
            const result = await botClient.startBot(config.agentWallet, baseAsset, quoteAsset, config.agentPrivateKey, quantity, side, "limit", // Default to limit orders
            spreadPercentage, referencePrice);
            return {
                content: [{
                        type: "text",
                        text: `Market maker bot started successfully:\n${JSON.stringify(result, null, 2)}`
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error starting market maker bot: ${err.message}`
                    }]
            };
        }
    });
    server.registerTool("stop_market_maker_bot", {
        title: "Stop Market Maker Bot",
        description: "Stop the running market maker bot and cancel all orders",
    }, async () => {
        try {
            const result = await botClient.stopBot();
            return {
                content: [{
                        type: "text",
                        text: `Market maker bot stopped successfully:\n${JSON.stringify(result, null, 2)}`
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error stopping market maker bot: ${err.message}`
                    }]
            };
        }
    });
    server.registerTool("get_market_maker_bot_status", {
        title: "Get Bot Status",
        description: "Get the current status and configuration of the market maker bot",
    }, async () => {
        try {
            const result = await botClient.getStatus();
            return {
                content: [{
                        type: "text",
                        text: `Market maker bot status:\n${JSON.stringify(result, null, 2)}`
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error getting bot status: ${err.message}`
                    }]
            };
        }
    });
    server.registerTool("modify_market_maker_bot_config", {
        title: "Modify Bot Configuration",
        description: "Modify the running bot's configuration (spread, quantity, reference price)",
        inputSchema: botModifySchema.shape,
    }, async ({ spreadPercentage, quantity, referencePrice }) => {
        try {
            const result = await botClient.modifyConfig(config.agentWallet, "SEI", // You might want to make these configurable
            "USDT", "bid", // You might want to track current side
            "limit", config.agentPrivateKey, spreadPercentage, quantity, referencePrice);
            return {
                content: [{
                        type: "text",
                        text: `Bot configuration modified successfully:\n${JSON.stringify(result, null, 2)}`
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error modifying bot configuration: ${err.message}`
                    }]
            };
        }
    });
    server.registerTool("force_register_orders", {
        title: "Force Register Orders",
        description: "Force the bot to register/update orders immediately",
    }, async () => {
        try {
            const result = await botClient.registerOrders(config.agentWallet, "SEI", // You might want to make these configurable
            "USDT", config.agentPrivateKey, "bid", // You might want to track current side
            "limit");
            return {
                content: [{
                        type: "text",
                        text: `Orders registered successfully:\n${JSON.stringify(result, null, 2)}`
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error registering orders: ${err.message}`
                    }]
            };
        }
    });
    server.registerTool("cancel_all_bot_orders", {
        title: "Cancel All Orders",
        description: "Cancel all current bot orders without stopping the bot",
    }, async () => {
        try {
            const result = await botClient.cancelOrders(config.agentWallet, "SEI", // You might want to make these configurable
            "USDT", config.agentPrivateKey, "bid", // You might want to track current side
            "limit");
            return {
                content: [{
                        type: "text",
                        text: `All orders cancelled successfully:\n${JSON.stringify(result, null, 2)}`
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error cancelling orders: ${err.message}`
                    }]
            };
        }
    });
    // ===== UTILITY TOOLS =====
    server.registerTool("get_market_price", {
        title: "Get Market Price",
        description: "Get current market price for a trading pair",
        inputSchema: z.object({
            baseAsset: z.string().describe("Base asset symbol"),
            quoteAsset: z.string().describe("Quote asset symbol"),
        }).shape,
    }, async ({ baseAsset, quoteAsset }) => {
        try {
            const price = await getMidPriceGate({ base: baseAsset, quote: quoteAsset });
            return {
                content: [{
                        type: "text",
                        text: `Current market price for ${baseAsset}/${quoteAsset}: ${price}`
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error getting market price: ${err.message}`
                    }]
            };
        }
    });
    // ===== COMBINED WORKFLOW TOOLS =====
    server.registerTool("start_market_making_workflow", {
        title: "Start Market Making Workflow",
        description: "Complete workflow: move assets from vault, start bot, begin market making",
        inputSchema: z.object({
            assetAmount: z.string().describe("Amount to move from vault"),
            baseAsset: z.string().describe("Base asset symbol"),
            quoteAsset: z.string().describe("Quote asset symbol"),
            quantity: z.number().describe("Order quantity"),
            side: z.enum(["bid", "ask"]).describe("Order side"),
            spreadPercentage: z.number().optional().describe("Spread percentage"),
            referencePrice: z.number().optional().describe("Reference price"),
        }).shape,
    }, async ({ assetAmount, baseAsset, quoteAsset, quantity, side, spreadPercentage, referencePrice }) => {
        try {
            const steps = [];
            // Step 1: Move assets from vault to wallet
            const client = await seiClientFactory();
            const moveResult = await client.callTool({
                name: "write_contract",
                arguments: {
                    contractAddress: config.vaultContractAddress,
                    abi: hyperfillOrderBookAbi,
                    functionName: "moveFromVaultToWallet",
                    args: [assetAmount, config.agentWallet],
                    network: "sei-testnet",
                }
            });
            steps.push(`✅ Moved ${assetAmount} assets from vault to wallet`);
            // Step 2: Start the market maker bot
            const botResult = await botClient.startBot(config.agentWallet, baseAsset, quoteAsset, config.agentPrivateKey, quantity, side, "limit", spreadPercentage || 0.5, referencePrice);
            steps.push(`✅ Started market maker bot for ${baseAsset}/${quoteAsset}`);
            return {
                content: [{
                        type: "text",
                        text: `Market making workflow completed successfully:\n\n${steps.join('\n')}\n\nBot Status: ${JSON.stringify(botResult, null, 2)}`
                    }]
            };
        }
        catch (err) {
            return {
                content: [{
                        type: "text",
                        text: `Error in market making workflow: ${err.message}`
                    }]
            };
        }
    });
}
