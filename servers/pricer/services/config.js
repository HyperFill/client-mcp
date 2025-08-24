import dotenv from 'dotenv';
dotenv.config();
export const config = {
    account: process.env.ACCOUNT_ADDRESS || "",
    agentPrivateKey: process.env.PRIVATE_KEY || "",
    agentWallet: process.env.ACCOUNT_ADDRESS || "",
    vaultContractAddress: process.env.VAULT_CONTRACT_ADDRESS || ""
};
