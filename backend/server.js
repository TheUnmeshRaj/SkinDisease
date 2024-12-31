import mongoose from "mongoose";

 const authSchema = new mongoose.Schema({
    username:String,
    message:String,
 })

 export const Auth = mongoose.model("Auth" , authSchema)