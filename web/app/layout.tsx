import "./globals.css";

export const metadata = {
  title: "WLASL Demo",
  description: "Local demo UI for offline inference."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
