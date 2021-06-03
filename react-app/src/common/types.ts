import {FileType} from "rsuite/es/Uploader";

export type Model = {
    id: number,
    name: string,
    description: string
}

export type AnalyzeData = {
    model: Model,
    images: FileType[]
}

export type ResultData = {
    images: string[],
    areas: number[]
}