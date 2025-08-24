// market-maker-bot-client.ts
import axios from 'axios';
import * as readline from 'readline';
class MarketMakerBotClient {
    constructor(botApiUrl = "http://localhost:8001") {
        this.apiUrl = botApiUrl;
    }
    async sendCommand(commandData) {
        try {
            const response = await axios.post(`${this.apiUrl}/bot/command`, commandData, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = response.data;
            console.log(data, "COMMADND");
            return data;
        }
        catch (error) {
            return {
                status: "error",
                message: error.response?.data?.message || error.message || "Unknown error"
            };
        }
    }
    async getStatus() {
        try {
            const response = await axios.get(`${this.apiUrl}/bot/status`);
            return response.data;
        }
        catch (error) {
            return {
                status: "error",
                message: error.response?.data?.message || error.message || "Unknown error"
            };
        }
    }
    async startBot(account, baseAsset, quoteAsset, privateKey, quantity, side, type, spreadPercentage = 0.5, referencePrice) {
        const command = {
            action: "start",
            account,
            base_asset: baseAsset,
            quote_asset: quoteAsset,
            private_key: privateKey,
            quantity,
            side,
            type,
            spread_percentage: spreadPercentage,
            reference_price: referencePrice
        };
        return this.sendCommand(command);
    }
    async stopBot() {
        const command = {
            action: "stop",
            side: "bid",
            type: "limit",
            account: "",
            base_asset: "",
            quote_asset: "",
            private_key: ""
        };
        return this.sendCommand(command);
    }
    async registerOrders(account, baseAsset, quoteAsset, privateKey, side, type) {
        const command = {
            action: "register",
            side,
            type,
            account,
            base_asset: baseAsset,
            quote_asset: quoteAsset,
            private_key: privateKey
        };
        return this.sendCommand(command);
    }
    async cancelOrders(account, baseAsset, quoteAsset, privateKey, side, type) {
        const command = {
            action: "cancel",
            account,
            side,
            type,
            base_asset: baseAsset,
            quote_asset: quoteAsset,
            private_key: privateKey
        };
        return this.sendCommand(command);
    }
    async modifyConfig(account, baseAsset, quoteAsset, side, type, privateKey, spreadPercentage, quantity, referencePrice) {
        const command = {
            action: "modify",
            account,
            side,
            type,
            base_asset: baseAsset,
            quote_asset: quoteAsset,
            private_key: privateKey,
            spread_percentage: spreadPercentage,
            quantity,
            reference_price: referencePrice
        };
        return this.sendCommand(command);
    }
}
// Interactive CLI implementation
class BotCLI {
    constructor() {
        this.defaultConfig = {
            account: "0xA548b3bbee2A5b779077234cc14b5c2CA3d95b85",
            baseAsset: "SEI",
            quoteAsset: "USDT",
            privateKey: "7e83358b55700eb111b17735faf8f3b4740c962cb3dc7ccfb75beb1171f7f596",
            quantity: 100.0,
            spreadPercentage: 0.5,
            side: "bid",
            type: "limit"
        };
        this.client = new MarketMakerBotClient();
        this.rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
    }
    question(prompt) {
        return new Promise((resolve) => {
            this.rl.question(prompt, resolve);
        });
    }
    printResult(result) {
        console.log(`Result: ${JSON.stringify(result, null, 2)}`);
    }
    showMenu() {
        console.log("\n=== Market Maker Bot Controller ===");
        console.log("Available commands:");
        console.log("1. start    - Start the market maker bot");
        console.log("2. stop     - Stop the market maker bot");
        console.log("3. status   - Get bot status");
        console.log("4. register - Force order registration");
        console.log("5. cancel   - Cancel all orders");
        console.log("6. modify   - Modify bot configuration");
        console.log("7. quit     - Exit");
        console.log();
    }
    async run() {
        this.showMenu();
        while (true) {
            try {
                const command = await this.question("Enter command: ");
                const cmd = command.trim().toLowerCase();
                if (cmd === "1" || cmd === "start") {
                    console.log("Starting bot with default config...");
                    const result = await this.client.startBot(this.defaultConfig.account, this.defaultConfig.baseAsset, this.defaultConfig.quoteAsset, this.defaultConfig.privateKey, this.defaultConfig.quantity, this.defaultConfig.side, this.defaultConfig.type, this.defaultConfig.spreadPercentage);
                    this.printResult(result);
                    //        account: string,
                    // baseAsset: string,
                    // quoteAsset: string,
                    // privateKey: string,
                    // quantity: number,
                    // side: string,
                    // spreadPercentage: number = 0.5,
                    // referencePrice?: number
                }
                else if (cmd === "2" || cmd === "stop") {
                    console.log("Stopping bot...");
                    const result = await this.client.stopBot();
                    this.printResult(result);
                }
                else if (cmd === "3" || cmd === "status") {
                    console.log("Getting bot status...");
                    const result = await this.client.getStatus();
                    this.printResult(result);
                }
                else if (cmd === "4" || cmd === "register") {
                    console.log("Forcing order registration...");
                    const result = await this.client.registerOrders(this.defaultConfig.account, this.defaultConfig.baseAsset, this.defaultConfig.quoteAsset, this.defaultConfig.privateKey, this.defaultConfig.side, this.defaultConfig.type);
                    this.printResult(result);
                }
                else if (cmd === "5" || cmd === "cancel") {
                    console.log("Canceling all orders...");
                    const result = await this.client.cancelOrders(this.defaultConfig.account, this.defaultConfig.baseAsset, this.defaultConfig.quoteAsset, this.defaultConfig.privateKey, this.defaultConfig.side, this.defaultConfig.type);
                    this.printResult(result);
                }
                else if (cmd === "6" || cmd === "modify") {
                    console.log("Modifying bot configuration...");
                    const spread = await this.question("Enter new spread percentage (or press Enter to skip): ");
                    const quantity = await this.question("Enter new quantity (or press Enter to skip): ");
                    const price = await this.question("Enter reference price (or press Enter to skip): ");
                    const result = await this.client.modifyConfig(this.defaultConfig.account, this.defaultConfig.baseAsset, this.defaultConfig.quoteAsset, this.defaultConfig.privateKey, this.defaultConfig.side, this.defaultConfig.type, spread.trim() ? parseFloat(spread) : undefined, quantity.trim() ? parseFloat(quantity) : undefined, price.trim() ? parseFloat(price) : undefined);
                    this.printResult(result);
                }
                else if (cmd === "7" || cmd === "quit") {
                    console.log("Exiting...");
                    break;
                }
                else {
                    console.log("Invalid command. Please try again.");
                }
            }
            catch (error) {
                console.log(`Error: ${error}`);
            }
        }
        this.rl.close();
    }
}
// Example usage as a module
export { MarketMakerBotClient };
// CLI runner
if (require.main === module) {
    const cli = new BotCLI();
    cli.run().catch(console.error);
}
