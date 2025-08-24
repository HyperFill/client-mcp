import { nanoid } from "nanoid";
import { ethers } from "ethers";
export class FilamentTrader {
    constructor(config) {
        // State management
        this._assets = new Map();
        this._positions = new Map();
        this._openOrders = new Map();
        this._balance = 0;
        this._lastUpdateTime = 0;
        this.baseUrl = config.baseUrl || "https://orderbook.filament.finance/sei";
        this.account = config.account.toLowerCase();
        this.privateKey = config.privateKey;
        this.defaultLeverage = config.defaultLeverage || 1.1;
        this.defaultSlippage = config.defaultSlippage || 5;
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
    // Utility Functions
    async handleOrderSignature(orderId) {
        const signer = new ethers.Wallet(this.privateKey);
        const orderSignature = await signer.signMessage(orderId);
        return orderSignature;
    }
    generateOrderId() {
        return nanoid().toLowerCase();
    }
    // Asset Management
    async fetchAssets() {
        const response = await fetch(`${this.baseUrl}/filament/api/v1/assets`);
        const assets = await response.json();
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
        const orderId = this.generateOrderId();
        const signature = await this.handleOrderSignature(orderId);
        const payload = {
            type: 'order',
            referralCode: null,
            orders: [
                {
                    account: this.account,
                    indexToken: assetInfo.indexToken,
                    orderId,
                    signature,
                    isBuy,
                    size,
                    leverage: leverage || this.defaultLeverage,
                    reduceOnly,
                    orderType: {
                        type: 'limit',
                        limit: {
                            tif: 'Gtc',
                            limitPrice: price,
                        },
                    },
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
        return { response, orderId };
    }
    async placeMarketOrder(asset, isBuy, size, leverage, slippage, reduceOnly = false) {
        const assetInfo = this.getAsset(asset);
        if (!assetInfo) {
            throw new Error(`Asset ${asset} not found. Please fetch assets first.`);
        }
        const orderId = this.generateOrderId();
        const signature = await this.handleOrderSignature(orderId);
        const payload = {
            type: 'order',
            referralCode: null,
            orders: [
                {
                    account: this.account,
                    indexToken: assetInfo.indexToken,
                    orderId,
                    signature,
                    isBuy,
                    size,
                    leverage: leverage || this.defaultLeverage,
                    reduceOnly,
                    orderType: {
                        type: 'trigger',
                        trigger: {
                            isMarket: true,
                            slippage: slippage || this.defaultSlippage,
                        },
                    },
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
        return { response, orderId };
    }
    async cancelOrder(orderId) {
        const signature = await this.handleOrderSignature(orderId);
        const payload = {
            type: 'cancel',
            cancels: [
                {
                    account: this.account,
                    orderId,
                    signature,
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
        // Remove from local state if successful
        if (response.ok) {
            this._openOrders.delete(orderId);
        }
        return response;
    }
    // Position Management
    async closePosition(asset, closePrice, quantity // If not provided, closes entire position
    ) {
        const position = this.getPosition(asset);
        if (!position) {
            throw new Error(`No open position found for ${asset}`);
        }
        // Determine order direction (opposite of position)
        const isBuy = position.side === 'SELL';
        const sizeToClose = quantity ? quantity * parseFloat(closePrice) : position.size;
        return this.placeLimitOrder(asset, isBuy, closePrice, sizeToClose, -1, true);
    }
    async addCollateral(asset, collateral, isBuy) {
        const orderId = this.generateOrderId();
        const signature = await this.handleOrderSignature(orderId);
        const payload = {
            type: 'updateIsolatedMargin',
            account: this.account,
            referralCode: null,
            asset,
            collateral,
            isBuy,
            isIncrement: true,
            orderId,
            signature,
            tpsl: 'na',
            size: '',
        };
        return fetch(`${this.baseUrl}/filament/api/v1/exchange`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
    }
    async removeCollateral(asset, collateral, isBuy) {
        const orderId = this.generateOrderId();
        const signature = await this.handleOrderSignature(orderId);
        const payload = {
            type: 'updateIsolatedMargin',
            account: this.account,
            referralCode: null,
            asset,
            collateral,
            isBuy,
            isIncrement: false,
            orderId,
            signature,
            tpsl: 'na',
            size: '',
        };
        return fetch(`${this.baseUrl}/filament/api/v1/exchange`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
    }
    async setTakeProfit(asset, takeProfitPrice, size, isBuy) {
        const orderId = this.generateOrderId();
        const takeProfitOrderId = this.generateOrderId();
        const signature = await this.handleOrderSignature(orderId);
        const takeProfitSignature = await this.handleOrderSignature(takeProfitOrderId);
        const payload = {
            type: 'updateIsolatedMargin',
            account: this.account,
            referralCode: null,
            asset,
            collateral: '',
            isBuy,
            isIncrement: true,
            orderId,
            signature,
            takeProfit: takeProfitPrice,
            takeProfitOrderId,
            takeProfitSignature,
            tpsl: 'tp',
            size,
        };
        return fetch(`${this.baseUrl}/filament/api/v1/exchange`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
    }
    async setStopLoss(asset, stopLossPrice, size, isBuy) {
        const orderId = this.generateOrderId();
        const stopLossOrderId = this.generateOrderId();
        const signature = await this.handleOrderSignature(orderId);
        const stopLossSignature = await this.handleOrderSignature(stopLossOrderId);
        const payload = {
            type: 'updateIsolatedMargin',
            account: this.account,
            referralCode: null,
            asset,
            collateral: '',
            isBuy,
            isIncrement: true,
            orderId,
            signature,
            stopLoss: stopLossPrice,
            stopLossOrderId,
            stopLossSignature,
            tpsl: 'sl',
            size,
        };
        return fetch(`${this.baseUrl}/filament/api/v1/exchange`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        });
    }
    // Data Retrieval Functions
    async fetchOpenOrders(asset, side, page = 0, size = 10) {
        let url = `${this.baseUrl}/v1/orders/open-orders/paginated/account/${this.account}?page=${page}&size=${size}`;
        if (asset)
            url += `&token=${asset}`;
        if (side)
            url += `&side=${side}`;
        const response = await fetch(url);
        const data = await response.json();
        // Update state
        if (page === 0)
            this._openOrders.clear(); // Clear on first page
        data.content.forEach((order) => {
            this._openOrders.set(order.orderId, order);
        });
        this._lastUpdateTime = Date.now();
        return data.content;
    }
    async fetchPositions(page = 0, size = 10) {
        const response = await fetch(`${this.baseUrl}/v1/positions/trades/paginated/${this.account}?page=${page}&size=${size}`);
        const data = await response.json();
        // Update state
        if (page === 0)
            this._positions.clear(); // Clear on first page
        data.content.forEach((position) => {
            this._positions.set(position.token, position);
        });
        this._lastUpdateTime = Date.now();
        return data.content;
    }
    async fetchBalance() {
        const response = await fetch(`${this.baseUrl}/v1/accountData/balance/${this.account}`);
        const balance = await response.json();
        this._balance = parseFloat(balance);
        this._lastUpdateTime = Date.now();
        return this._balance;
    }
    async fetchTradeHistory(page = 0, size = 10, associatedOrderId) {
        let url = `${this.baseUrl}/v1/positions/tradeHistory/paginated/${this.account}?page=${page}&size=${size}`;
        if (associatedOrderId)
            url += `&associatedOrderId=${associatedOrderId}`;
        const response = await fetch(url);
        const data = await response.json();
        return data.content;
    }
    async getOrderStatus(orderIds) {
        const response = await fetch(`${this.baseUrl}/v1/orders/latest-status`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(orderIds),
        });
        const orders = await response.json();
        // Update local state
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
    // WebSocket connection helpers (you can implement these based on the documentation)
    connectOrderBookWebSocket(asset) {
        const assetInfo = this.getAsset(asset);
        if (!assetInfo) {
            throw new Error(`Asset ${asset} not found`);
        }
        // WebSocket implementation would go here
        // Based on the documentation's WebSocket examples
        throw new Error('WebSocket implementation needed');
    }
    connectOrderUpdatesWebSocket() {
        // WebSocket implementation for order updates would go here
        throw new Error('WebSocket implementation needed');
    }
}
// Export the class and types
export default FilamentTrader;
