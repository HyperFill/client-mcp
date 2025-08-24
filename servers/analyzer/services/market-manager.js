import HyperFillMMClient from "../client/hyper-fillmm-client";
export class MarketManager {
    constructor() {
        this._supportedMarkets = [];
        this._marketList = [
            {
                marketName: "hyperfill",
                id: "123"
            }
        ];
    }
    getMarketList() {
        return this._marketList;
    }
    getMarketClient(marketName) {
        let client;
        for (let market of this._marketList) {
            if (market.marketName == marketName) {
                let config = {
                    baseUrl: "http://localhost:3000",
                    privateKey: "0x",
                    simulationMode: true,
                    account: "1"
                };
                client = new HyperFillMMClient(config);
            }
        }
        return client;
    }
}
