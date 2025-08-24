import { nanoid } from "nanoid";
export class HyperFillMMClient {
    constructor(config) {
        // State management
        this._assets = new Map();
        this._positions = new Map();
        this._openOrders = new Map();
        this._balance = 0;
        this._lastUpdateTime = 0;
        // Simulation data storage
        this._simulatedOrders = new Map();
        this._simulatedPositions = new Map();
        this._simulatedTradeHistory = [];
        this.baseUrl = config.baseUrl || "https://orderbook.hyperfill.finance/sei";
        this.account = config.account.toLowerCase();
        this.privateKey = config.privateKey;
        // this.defaultLeverage = config.defaultLeverage || 1.1;
        this.simulationMode = config.simulationMode || true;
        // Initialize simulation data
        if (this.simulationMode) {
            this.initializeSimulationData();
        }
    }
    // Initialize dummy data for simulation
    initializeSimulationData() {
        // Add some dummy open orders
        const dummyOrder1 = {
            orderId: nanoid(),
            account: this.account,
            token: "SEI/USDT",
            size: 1000,
            // collateral: 100,
            side: 'BUY',
            typeOfOrder: 'Limit',
            triggerPrice: 0.315,
            orderStatus: 'OPEN',
            orderValue: 315,
            openTime: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
            decreaseOrder: false,
            isStopLoss: false,
            isTakeProfit: false,
        };
        const dummyOrder2 = {
            orderId: nanoid(),
            account: this.account,
            token: "BTC/USDT",
            size: 0.1,
            // collateral: 500,
            side: 'SELL',
            typeOfOrder: 'Limit',
            triggerPrice: 45500,
            orderStatus: 'OPEN',
            orderValue: 4550,
            openTime: new Date(Date.now() - 1800000).toISOString(), // 30 min ago
            decreaseOrder: false,
            isStopLoss: false,
            isTakeProfit: false,
        };
        this._simulatedOrders.set(dummyOrder1.orderId, dummyOrder1);
        this._simulatedOrders.set(dummyOrder2.orderId, dummyOrder2);
        // Add some dummy positions
        const dummyPosition1 = {
            openTime: new Date(Date.now() - 7200000).toISOString(), // 2 hours ago
            account: this.account,
            token: "ETH/USDT",
            side: 'BUY',
            entryPrice: 2450,
            markPrice: 2465,
            size: 2,
            quantity: 2,
            collateral: 500,
            fees: 2.45,
            realizedPnL: 30,
            takeProfit: null,
            stopLoss: 2400,
            liquidationPrice: 2350,
            leverage: 10,
        };
        this._simulatedPositions.set("ETH/USDT", dummyPosition1);
        // Add some trade history
        this._simulatedTradeHistory = [
            {
                id: nanoid(),
                tradeTime: new Date(Date.now() - 3600000).toISOString(),
                account: this.account,
                token: "SEI/USDT",
                side: 'BUY',
                price: 0.318,
                size: 500,
                tradeValue: 159,
                fee: 0.159,
                closedPnL: 0,
                orderType: "Market",
                associatedOrderId: nanoid(),
            },
            {
                id: nanoid(),
                tradeTime: new Date(Date.now() - 7200000).toISOString(),
                account: this.account,
                token: "ETH/USDT",
                side: 'BUY',
                price: 2450,
                size: 2,
                tradeValue: 4900,
                fee: 4.9,
                closedPnL: 0,
                orderType: "Limit",
                associatedOrderId: nanoid(),
            }
        ];
        // Set initial balance
        this._balance = 10000;
    }
    // Generate realistic dummy assets
    generateDummyAssets() {
        const assets = [
            {
                id: "1",
                assetName: "SEI/USDT",
                indexToken: "1",
                tickPrice: 0.32121,
                bidPrice: 0.32119,
                askPrice: 0.3225,
                tradingEnabled: true,
            },
            {
                id: "2",
                assetName: "BTC/USDT",
                indexToken: "2",
                tickPrice: 45234.56,
                bidPrice: 45230.12,
                askPrice: 45245.78,
                tradingEnabled: true,
            },
            {
                id: "3",
                assetName: "ETH/USDT",
                indexToken: "3",
                tickPrice: 2465.34,
                bidPrice: 2464.12,
                askPrice: 2466.89,
                tradingEnabled: true,
            },
            {
                id: "4",
                assetName: "SOL/USDT",
                indexToken: "4",
                tickPrice: 89.45,
                bidPrice: 89.42,
                askPrice: 89.67,
                tradingEnabled: true,
            },
            {
                id: "5",
                assetName: "AVAX/USDT",
                indexToken: "5",
                tickPrice: 34.78,
                bidPrice: 34.76,
                askPrice: 34.82,
                tradingEnabled: false, // Disabled for testing
            }
        ];
        return assets;
    }
    // Getters for state
    get assets() {
        return Array.from(this._assets.values());
    }
    get positions() {
        return Array.from(this._positions.values());
    }
    get openOrders() {
        return Array.from(this._openOrders.values());
    }
    get balance() {
        return this._balance;
    }
    get lastUpdateTime() {
        return new Date(this._lastUpdateTime);
    }
    // Asset Management
    async fetchAssets() {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 100));
        let assets;
        if (this.simulationMode) {
            assets = this.generateDummyAssets();
        }
        else {
            const response = await fetch(`${this.baseUrl}/filament/api/v1/assets`);
            assets = await response.json();
        }
        // Update state
        this._assets.clear();
        assets.forEach(asset => {
            this._assets.set(asset.assetName, asset);
        });
        this._lastUpdateTime = Date.now();
        return assets;
    }
    getAsset(symbol) {
        return this._assets.get(symbol);
    }
    getAssetByIndexToken(indexToken) {
        return Array.from(this._assets.values()).find(asset => asset.indexToken === indexToken);
    }
    // Order Placement Functions
    async placeLimitOrder(asset, isBuy, price, size, leverage, reduceOnly = false) {
        const assetInfo = this.getAsset(asset);
        if (!assetInfo) {
            throw new Error(`Asset ${asset} not found. Please fetch assets first.`);
        }
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 200));
        const orderId = nanoid();
        const order = {
            account: this.account,
            indexToken: assetInfo.indexToken,
            orderId,
            signature: "dummy_signature_" + orderId,
            isBuy,
            size,
            leverage: leverage,
            reduceOnly,
            orderType: {
                type: 'limit',
                limit: {
                    tif: 'Gtc',
                    limitPrice: price,
                },
            },
        };
        if (this.simulationMode) {
            // Create order status for simulation
            const orderStatus = {
                orderId,
                account: this.account,
                token: asset,
                size,
                // collateral: size * parseFloat(price),
                side: isBuy ? 'BUY' : 'SELL',
                typeOfOrder: 'Limit',
                triggerPrice: parseFloat(price),
                orderStatus: 'OPEN',
                orderValue: size * parseFloat(price),
                openTime: new Date().toISOString(),
                decreaseOrder: reduceOnly,
                isStopLoss: false,
                isTakeProfit: false,
            };
            this._simulatedOrders.set(orderId, orderStatus);
            this._openOrders.set(orderId, orderStatus);
            return { response: { success: true, orderId, order } };
        }
        const payload = {
            type: 'order',
            referralCode: null,
            orders: [order]
        };
        const response = await fetch(`${this.baseUrl}/hyperfill/api/v1/exchange`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
        return { response: await response.json() };
    }
    async placeMarketOrder(asset, isBuy, size, leverage, slippage, reduceOnly = false) {
        const assetInfo = this.getAsset(asset);
        if (!assetInfo) {
            throw new Error(`Asset ${asset} not found. Please fetch assets first.`);
        }
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 150));
        const orderId = nanoid();
        const order = {
            account: this.account,
            indexToken: assetInfo.indexToken,
            orderId,
            signature: "dummy_signature_" + orderId,
            isBuy,
            size,
            // leverage: leverage || this.defaultLeverage,
            reduceOnly,
            orderType: {
                type: 'trigger',
                trigger: {
                    isMarket: true,
                    slippage: slippage || 5,
                },
            },
        };
        if (this.simulationMode) {
            // Market orders execute immediately in simulation
            const executionPrice = isBuy ? assetInfo.askPrice : assetInfo.bidPrice;
            // Create trade history entry
            const trade = {
                id: nanoid(),
                tradeTime: new Date().toISOString(),
                account: this.account,
                token: asset,
                side: isBuy ? 'BUY' : 'SELL',
                price: executionPrice,
                size,
                tradeValue: size * executionPrice,
                fee: (size * executionPrice) * 0.001, // 0.1% fee
                closedPnL: 0,
                orderType: "Market",
                associatedOrderId: orderId,
            };
            this._simulatedTradeHistory.unshift(trade);
            // Update balance
            const cost = trade.tradeValue + trade.fee;
            this._balance -= isBuy ? cost : -cost;
            return {
                response: {
                    success: true,
                    orderId,
                    executedPrice: executionPrice,
                    executedSize: size,
                    trade
                }
            };
        }
        const payload = {
            type: 'order',
            referralCode: null,
            orders: [order],
        };
        const response = await fetch(`${this.baseUrl}/filament/api/v1/exchange`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
        return { response: await response.json() };
    }
    async cancelOrder(orderId) {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 100));
        if (this.simulationMode) {
            const order = this._simulatedOrders.get(orderId);
            if (order) {
                order.orderStatus = 'CANCELLED';
                order.closedTime = new Date().toISOString();
                this._openOrders.delete(orderId);
                return true;
            }
            return false;
        }
        const payload = {
            type: 'cancel',
            cancels: [
                {
                    account: this.account,
                    orderId,
                    signature: "dummy_signature_" + orderId,
                },
            ],
        };
        const response = await fetch(`${this.baseUrl}/filament/api/v1/exchange`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
        if (response.ok) {
            this._openOrders.delete(orderId);
            return true;
        }
        return false;
    }
    // Position Management
    async closePosition(asset, closePrice, quantity) {
        const position = this.getPosition(asset);
        if (!position) {
            throw new Error(`No open position found for ${asset}`);
        }
        const isBuy = position.side === 'SELL';
        const sizeToClose = quantity ? quantity * parseFloat(closePrice) : position.size;
        return this.placeLimitOrder(asset, isBuy, closePrice, sizeToClose, undefined, true);
    }
    // Data Retrieval Functions
    async fetchOpenOrders(asset, side, page = 0, size = 10) {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 50));
        if (this.simulationMode) {
            let orders = Array.from(this._simulatedOrders.values())
                .filter(order => order.orderStatus === 'OPEN');
            if (asset) {
                orders = orders.filter(order => order.token === asset);
            }
            if (side) {
                orders = orders.filter(order => order.side === side);
            }
            // Simulate pagination
            const start = page * size;
            const paginatedOrders = orders.slice(start, start + size);
            // Update state
            if (page === 0)
                this._openOrders.clear();
            paginatedOrders.forEach(order => {
                this._openOrders.set(order.orderId, order);
            });
            this._lastUpdateTime = Date.now();
            return paginatedOrders;
        }
        let url = `${this.baseUrl}/v1/orders/open-orders/paginated/account/${this.account}?page=${page}&size=${size}`;
        if (asset)
            url += `&token=${asset}`;
        if (side)
            url += `&side=${side}`;
        const response = await fetch(url);
        const data = await response.json();
        if (page === 0)
            this._openOrders.clear();
        data.content.forEach((order) => {
            this._openOrders.set(order.orderId, order);
        });
        this._lastUpdateTime = Date.now();
        return data.content;
    }
    async fetchPositions(page = 0, size = 10) {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 50));
        if (this.simulationMode) {
            const positions = Array.from(this._simulatedPositions.values());
            // Simulate pagination
            const start = page * size;
            const paginatedPositions = positions.slice(start, start + size);
            // Update state
            if (page === 0)
                this._positions.clear();
            paginatedPositions.forEach(position => {
                this._positions.set(position.token, position);
            });
            this._lastUpdateTime = Date.now();
            return paginatedPositions;
        }
        const response = await fetch(`${this.baseUrl}/v1/positions/trades/paginated/${this.account}?page=${page}&size=${size}`);
        const data = await response.json();
        if (page === 0)
            this._positions.clear();
        data.content.forEach((position) => {
            this._positions.set(position.token, position);
        });
        this._lastUpdateTime = Date.now();
        return data.content;
    }
    async fetchBalance() {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 30));
        if (this.simulationMode) {
            // Add some random fluctuation to simulate market changes
            const fluctuation = (Math.random() - 0.5) * 10;
            this._balance += fluctuation;
            this._lastUpdateTime = Date.now();
            return this._balance;
        }
        const response = await fetch(`${this.baseUrl}/v1/accountData/balance/${this.account}`);
        const balance = await response.json();
        this._balance = parseFloat(balance);
        this._lastUpdateTime = Date.now();
        return this._balance;
    }
    async fetchTradeHistory(page = 0, size = 10, associatedOrderId) {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 50));
        if (this.simulationMode) {
            let trades = [...this._simulatedTradeHistory];
            if (associatedOrderId) {
                trades = trades.filter(trade => trade.associatedOrderId === associatedOrderId);
            }
            // Simulate pagination
            const start = page * size;
            return trades.slice(start, start + size);
        }
        let url = `${this.baseUrl}/v1/positions/tradeHistory/paginated/${this.account}?page=${page}&size=${size}`;
        if (associatedOrderId)
            url += `&associatedOrderId=${associatedOrderId}`;
        const response = await fetch(url);
        const data = await response.json();
        return data.content;
    }
    async getOrderStatus(orderIds) {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 50));
        if (this.simulationMode) {
            const orders = orderIds
                .map(id => this._simulatedOrders.get(id))
                .filter(order => order !== undefined);
            // Update local state
            orders.forEach(order => {
                if (order.orderStatus === 'OPEN') {
                    this._openOrders.set(order.orderId, order);
                }
                else {
                    this._openOrders.delete(order.orderId);
                }
            });
            return orders;
        }
        const response = await fetch(`${this.baseUrl}/v1/orders/latest-status`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(orderIds),
        });
        const orders = await response.json();
        orders.forEach((order) => {
            if (order.orderStatus === 'OPEN') {
                this._openOrders.set(order.orderId, order);
            }
            else {
                this._openOrders.delete(order.orderId);
            }
        });
        return orders;
    }
    // State Query Functions
    getPosition(asset) {
        return this._positions.get(asset);
    }
    getOrder(orderId) {
        return this._openOrders.get(orderId);
    }
    hasOpenPosition(asset) {
        return this._positions.has(asset);
    }
    hasOpenOrders(asset) {
        if (!asset)
            return this._openOrders.size > 0;
        return Array.from(this._openOrders.values()).some(order => order.token === asset);
    }
    // Utility Functions
    async refreshAllData() {
        await Promise.all([
            this.fetchAssets(),
            this.fetchPositions(),
            this.fetchOpenOrders(),
            this.fetchBalance(),
        ]);
    }
    // Simulation specific methods
    toggleSimulationMode(enabled) {
        this.simulationMode = enabled;
        if (enabled) {
            this.initializeSimulationData();
        }
    }
    // Method to simulate order execution (useful for testing)
    simulateOrderExecution(orderId, executionPrice) {
        if (!this.simulationMode)
            return false;
        const order = this._simulatedOrders.get(orderId);
        if (!order || order.orderStatus !== 'OPEN')
            return false;
        const asset = this.getAsset(order.token);
        if (!asset)
            return false;
        const price = executionPrice || (order.side === 'BUY' ? asset.askPrice : asset.bidPrice);
        // Update order status
        order.orderStatus = 'MATCHED';
        order.closedTime = new Date().toISOString();
        this._openOrders.delete(orderId);
        // Create trade history
        const trade = {
            id: nanoid(),
            tradeTime: new Date().toISOString(),
            account: this.account,
            token: order.token,
            side: order.side,
            price,
            size: order.size,
            tradeValue: order.size * price,
            fee: (order.size * price) * 0.001,
            closedPnL: 0,
            orderType: order.typeOfOrder,
            associatedOrderId: orderId,
        };
        this._simulatedTradeHistory.unshift(trade);
        // Update balance
        const cost = trade.tradeValue + trade.fee;
        this._balance -= order.side === 'BUY' ? cost : -cost;
        return true;
    }
}
export default HyperFillMMClient;
