// web/src/app/countries/[code]/page.tsx
import { CountryClientView } from "./CountryClientView";

const LAUNCH_CODES = ["us", "ca", "br", "gb", "de", "jp", "cn", "in", "kr", "au"];

export function generateStaticParams() {
  return LAUNCH_CODES.map((code) => ({ code }));
}

interface PageProps {
  params: { code: string };
}

export default function CountryPage({ params }: PageProps) {
  return <CountryClientView code={params.code} />;
}
