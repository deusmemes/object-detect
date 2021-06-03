import axios, {AxiosInstance, AxiosPromise} from 'axios';
import {URLS} from "../common/enums";
import {AnalyzeData} from "../common/types";

export const loadImages = () => {

}

export class HttpService {
    private static instance: HttpService;
    private ax: AxiosInstance;

    private constructor() {
        this.ax = axios.create({
            baseURL: 'http://127.0.0.1:5000'
        });
    }

    public static getInstance(): HttpService {
        if (!this.instance) {
            this.instance = new HttpService()
        }

        return this.instance;
    }

    public analyze(requestData: AnalyzeData): AxiosPromise {
        console.log(requestData)
        const formData = new FormData();
        requestData.images.forEach(file => formData.append('files', file.blobFile!!, file.name))
        formData.append('model', requestData.model.name)
        console.log(formData)

        return this.ax.post(URLS.ANALYZE, formData, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
    }

    public getModels(): AxiosPromise {
        return this.ax.get('/models')
    }
}