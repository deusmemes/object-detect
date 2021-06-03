import {Notification} from "rsuite";

export class Message {
    public static success(text: string, header: string) {
        Notification.success({
            title: header,
            description: text
          });
    }
}